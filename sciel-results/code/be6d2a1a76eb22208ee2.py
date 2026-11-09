import time
import numpy as np
from scipy.optimize import linprog, minimize
import itertools

_TOL = 1e-7
_DEADLINE = 630.0

def _validate(centers, sides, n=10):
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False, 0.0
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False, 0.0
    if np.any(sides <= 0):
        return False, 0.0
    for i in range(n):
        if centers[i, 0] - sides[i] / 2 < -_TOL or centers[i, 0] + sides[i] / 2 > 1 + _TOL:
            return False, 0.0
        if centers[i, 1] - sides[i] / 2 < -_TOL or centers[i, 1] + sides[i] / 2 > 1 + _TOL:
            return False, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            t = (sides[i] + sides[j]) / 2 - _TOL
            if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                return False, 0.0
    value = float(np.sum(sides))
    return True, value

def _solve_lp_for_pattern(n, pattern):
    """
    Solve the LP for a given separation pattern.
    Variables x = [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
    Objective: maximize sum(s_i) -> minimize -sum(s_i)
    Constraints:
    1. 0 <= cx_i - s_i/2
    2. cx_i + s_i/2 <= 1
    3. 0 <= cy_i - s_i/2
    4. cy_i + s_i/2 <= 1
    5. Non-overlap based on pattern[i,j]
       If pattern[i,j] == 0 (x-separation): 
           Assume cx_i <= cx_j for simplicity? No, we must handle both or fix order.
           Actually, a "pattern" usually implies a specific geometric configuration.
           To make the LP linear, we must fix the relative order or separation axis AND direction.
           Let's define pattern[i,j] in {0, 1, 2, 3}:
           0: i is left of j (cx_j - cx_i >= (s_i+s_j)/2)
           1: i is right of j (cx_i - cx_j >= (s_i+s_j)/2)
           2: i is below j (cy_j - cy_i >= (s_i+s_j)/2)
           3: i is above j (cy_i - cy_j >= (s_i+s_j)/2)
           
           Note: Enumerating 4^(45) is impossible. 
           However, we can use a simpler heuristic: 
           We know that in optimal packings, often a grid-like structure exists.
           Instead of enumerating all 4^45, we will generate specific patterns based on a "grid" hypothesis 
           or use a smaller set of candidate patterns derived from a coarse grid solution.
           
           Alternative: The inspiration suggests enumerating patterns. 
           But 4^45 is too large. 
           Let's look at the inspiration again: "enumerate separation patterns... solve LP".
           Perhaps the number of pairs is small enough for a different encoding? No, 45 pairs.
           
           Wait, the inspiration score is 1.2. How did they do it?
           Maybe they didn't enumerate ALL 4^45. 
           Maybe they used a different formulation.
           
           Let's try a different structural change:
           Use a **Penalty Method with Simulated Annealing** or **Differential Evolution** 
           to find a good feasible point, then use SLSQP to polish.
           The current program uses SLSQP from static grid guesses.
           The stagnation suggests the grid guesses are not good enough for SLSQP to escape local minima.
           
           Let's try **Differential Evolution** (global optimizer) to find a good feasible point, 
           then refine with SLSQP.
           
           Why DE?
           - It is a global optimizer.
           - It handles non-convex problems.
           - We can penalize violations in the objective function.
           
           Objective for DE: -sum(sides) + penalty(violations)
           Then take the best feasible result and polish with SLSQP.
    """
    pass

def squares_in_square10() -> tuple:
    n = 10
    t0 = time.time()
    deadline = t0 + _DEADLINE - 30.0

    best_value = 0.0
    best_centers = None
    best_sides = None

    # Fallback grid
    def _get_grid():
        k = 4
        side = 1.0 / k
        idx_arr = np.arange(n)
        cols = idx_arr % k
        rows = idx_arr // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    best_centers, best_sides = _get_grid()
    valid, best_value = _validate(best_centers, best_sides, n)

    # Strategy: Differential Evolution with Penalty
    # x = [cx_0, cy_0, s_0, ...]
    
    def _penalty_objective(x):
        centers = np.column_stack([x[2*np.arange(n)], x[2*np.arange(n)+1]])
        sides = x[2*np.arange(n)+2]
        
        # Objective: minimize -sum(sides)
        obj = -np.sum(sides)
        
        # Penalties
        pen = 0.0
        
        # Boundary penalties
        for i in range(n):
            if centers[i, 0] - sides[i]/2 < 0:
                pen += (centers[i, 0] - sides[i]/2)**2
            if centers[i, 0] + sides[i]/2 > 1:
                pen += (centers[i, 0] + sides[i]/2 - 1)**2
            if centers[i, 1] - sides[i]/2 < 0:
                pen += (centers[i, 1] - sides[i]/2)**2
            if centers[i, 1] + sides[i]/2 > 1:
                pen += (centers[i, 1] + sides[i]/2 - 1)**2
        
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2
                dx = abs(centers[i, 0] - centers[j, 0])
                dy = abs(centers[i, 1] - centers[j, 1])
                # We want dx >= t OR dy >= t
                # min(dx-t, dy-t) should be >= 0
                # If min < 0, penalize
                sep = min(dx - t, dy - t)
                if sep < 0:
                    pen += sep**2
                    
        return obj + 1000 * pen

    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
        bounds.append((0.01, 1.0)) # s

    # Use scipy.optimize.differential_evolution
    from scipy.optimize import differential_evolution

    try:
        # Run DE for a while
        result_de = differential_evolution(
            _penalty_objective, 
            bounds=bounds, 
            seed=42, 
            maxiter=100, 
            tol=1e-6,
            popsize=15,
            init='latinhypercube',
            polish=False # We will polish manually
        )
        
        if result_de.success or True:
            x_de = result_de.x
            centers_de = np.column_stack([x_de[2*np.arange(n)], x_de[2*np.arange(n)+1]])
            sides_de = x_de[2*np.arange(n)+2]
            
            valid_de, val_de = _validate(centers_de, sides_de, n)
            if valid_de and val_de > best_value:
                best_value = val_de
                best_centers = centers_de.copy()
                best_sides = sides_de.copy()
                
    except Exception as e:
        pass

    # If we have a good candidate, try to polish with SLSQP
    if best_centers is not None and time.time() < deadline - 60:
        # We need a smooth objective and constraints for SLSQP
        # Let's reuse the SLSQP logic from the current program but start from the DE result
        
        def _objective(x, n=10):
            sides = x[2 * np.arange(n) + 2]
            return -np.sum(sides)

        def _grad_boundary(x, n, i, which):
            grad = np.zeros(3 * n)
            ci, cji, si = 2 * i, 2 * i + 1, 2 * i + 2
            if which == 'left':
                grad[ci] = 1.0
                grad[si] = -0.5
            elif which == 'right':
                grad[ci] = -1.0
                grad[si] = -0.5
            elif which == 'bottom':
                grad[cji] = 1.0
                grad[si] = -0.5
            elif which == 'top':
                grad[cji] = -1.0
                grad[si] = -0.5
            return grad

        def _make_nonoverlap_constraint(i, j, n):
            ci, cji, si = 2 * i, 2 * i + 1, 2 * i + 2
            cj, cjj, sj = 2 * j, 2 * j + 1, 2 * j + 2

            def fun(x):
                dx = x[ci] - x[cj]
                dy = x[cji] - x[cjj]
                a = abs(dx) - (x[si] + x[sj]) / 2.0
                b = abs(dy) - (x[si] + x[sj]) / 2.0
                return (a + b - abs(a - b)) / 2.0

            def jac(x):
                dx = x[ci] - x[cj]
                dy = x[cji] - x[cjj]
                a = abs(dx) - (x[si] + x[sj]) / 2.0
                b = abs(dy) - (x[si] + x[sj]) / 2.0
                grad = np.zeros(3 * n)
                
                diff = a - b
                if abs(diff) < 1e-12:
                    da_da = 0.5
                    db_da = 0.5
                else:
                    sgn = 1.0 if diff > 0 else -1.0
                    da_da = 0.5 * (1.0 - sgn)
                    db_da = 0.5 * (1.0 + sgn)

                sgn_dx = 1.0 if dx >= 0 else -1.0
                sgn_dy = 1.0 if dy >= 0 else -1.0

                grad[ci] = da_da * sgn_dx
                grad[cj] = da_da * (-sgn_dx)
                grad[si] = da_da * (-0.5)
                grad[sj] = da_da * (-0.5)

                grad[cji] = db_da * sgn_dy
                grad[cjj] = db_da * (-sgn_dy)
                grad[si] += db_da * (-0.5)
                grad[sj] += db_da * (-0.5)

                return grad

            return {'type': 'ineq', 'fun': fun, 'jac': jac}

        cons = []
        for i in range(n):
            ci, cji, si = 2 * i, 2 * i + 1, 2 * i + 2
            cons.append({'type': 'ineq', 'fun': lambda x, ci=ci, si=si: x[ci] - x[si] / 2.0, 'jac': lambda x, ci=ci, si=si: _grad_boundary(x, n, i, 'left')})
            cons.append({'type': 'ineq', 'fun': lambda x, ci=ci, si=si: 1.0 - x[ci] - x[si] / 2.0, 'jac': lambda x, ci=ci, si=si: _grad_boundary(x, n, i, 'right')})
            cons.append({'type': 'ineq', 'fun': lambda x, cji=cji, si=si: x[cji] - x[si] / 2.0, 'jac': lambda x, cji=cji, si=si: _grad_boundary(x, n, i, 'bottom')})
            cons.append({'type': 'ineq', 'fun': lambda x, cji=cji, si=si: 1.0 - x[cji] - x[si] / 2.0, 'jac': lambda x, cji=cji, si=si: _grad_boundary(x, n, i, 'top')})

        for i in range(n):
            for j in range(i + 1, n):
                cons.append(_make_nonoverlap_constraint(i, j, n))

        x0 = np.zeros(3 * n)
        x0[2 * np.arange(n)] = best_centers[:, 0]
        x0[2 * np.arange(n) + 1] = best_centers[:, 1]
        x0[2 * np.arange(n) + 2] = best_sides

        bounds_slsqp = []
        for i in range(n):
            bounds_slsqp.append((0.0, 1.0))
            bounds_slsqp.append((0.0, 1.0))
            bounds_slsqp.append((1e-9, 1.0))

        try:
            res = minimize(_objective, x0, args=(n,), method='SLSQP',
                           bounds=bounds_slsqp, constraints=cons,
                           options={'maxiter': 2000, 'ftol': 1e-12})
            
            if res.success:
                x_sol = res.x
                centers_sol = np.column_stack([x_sol[2 * np.arange(n)], x_sol[2 * np.arange(n) + 1]]).astype(np.float64)
                sides_sol = x_sol[2 * np.arange(n) + 2].astype(np.float64)
                
                valid_sol, val_sol = _validate(centers_sol, sides_sol, n)
                if valid_sol and val_sol > best_value:
                    best_value = val_sol
                    best_centers = centers_sol.copy()
                    best_sides = sides_sol.copy()
        except Exception:
            pass

    # Final validation
    valid, _ = _validate(best_centers, best_sides, n)
    if not valid:
        best_centers, best_sides = _get_grid()

    return best_centers, best_sides
