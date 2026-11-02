import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Delaunay

N = 21

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    best_value = 0.0
    best_result = None

    def _is_valid(cx, cy, r, W, H, tol=1e-9):
        if W <= 0.01 or H <= 0.01:
            return False
        if np.any(cx - r < -tol) or np.any(cy - r < -tol):
            return False
        if np.any(cx + r > W + tol) or np.any(cy + r > H + tol):
            return False
        # Vectorized pairwise check
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        d2 = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        # Check only upper triangle
        iu = np.triu_indices(N, k=1)
        if np.any(d2[iu] < r_sum[iu]**2 - tol):
            return False
        return True

    def _objective(z, beta=50.0):
        """
        Objective function with smooth penalties.
        z = [x1, y1, r1, ..., xN, yN, rN, w]
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        h = 2.0 - w

        if w < 0.2 or h < 0.2 or np.any(r < 0):
            return 1e6

        obj = -np.sum(r)

        # Smooth overlap penalty
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d_sq = dx * dx + dy * dy
        r_sum = r[:, None] + r[None, :]
        
        iu = np.triu_indices(N, k=1)
        d_pairs = np.sqrt(np.maximum(d_sq[iu], 1e-12))
        overlap = r_sum[iu] - d_pairs
        # Soft penalty: sum of exp(beta * overlap) for overlapping pairs
        # This is smooth and pushes apart strongly
        overlap_penalty = np.sum(np.exp(beta * np.maximum(overlap, 0.0)))
        obj += 10.0 * overlap_penalty

        # Boundary penalties
        left_viol = np.maximum(r - x, 0.0)
        right_viol = np.maximum(x + r - w, 0.0)
        bottom_viol = np.maximum(r - y, 0.0)
        top_viol = np.maximum(y + r - h, 0.0)
        
        boundary_penalty = np.sum(np.exp(beta * left_viol)) + \
                           np.sum(np.exp(beta * right_viol)) + \
                           np.sum(np.exp(beta * bottom_viol)) + \
                           np.sum(np.exp(beta * top_viol))
        obj += 10.0 * boundary_penalty

        return obj

    def _gradient_descent(z, time_budget):
        """
        Perform gradient descent to polish the solution.
        Uses numerical gradients for simplicity and robustness, or analytic if possible.
        Given time constraints, numerical is safer but slower. 
        However, for 64 variables, numerical gradient (64 evals) is feasible.
        """
        z = z.copy()
        step_size = 0.01
        min_step = 1e-6
        max_iter = 2000
        last_obj = _objective(z)
        
        for i in range(max_iter):
            if time.time() > time_budget:
                break
            
            # Compute gradient numerically
            grad = np.zeros_like(z)
            eps = 1e-5
            for j in range(len(z)):
                z_plus = z.copy()
                z_plus[j] += eps
                grad[j] = (_objective(z_plus) - last_obj) / eps
            
            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-8:
                break
            
            grad_normed = grad / grad_norm
            step = step_size
            
            # Try step
            z_new = z - step * grad_normed
            
            # Simple line search
            obj_new = _objective(z_new)
            if obj_new < last_obj:
                z = z_new
                last_obj = obj_new
                step_size = min(step_size * 1.1, 0.05)
            else:
                step_size *= 0.5
                if step_size < min_step:
                    break
                    
        return z

    def _slsqp_polish(z, time_budget):
        """
        Final SLSQP polish with constraints.
        """
        x = z[0::3][:N]
        y = z[1::3][:N]
        r = z[2::3][:N]
        w = z[-1]
        
        def obj(zv):
            rv = zv[2::3][:N]
            return -np.sum(rv)

        constraints = []
        bounds = []
        
        for i in range(N):
            bounds.extend([(0, 2), (0, 2), (0.001, 1.0)])
        bounds.append((0.3, 1.7))

        # Boundary constraints
        for i in range(N):
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[-1] - p[3*i] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]})
            constraints.append({'type': 'ineq', 'fun': lambda p, i=i: (2.0 - p[-1]) - p[3*i+1] - p[3*i+2]})

        # Overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                    (p[3*i] - p[3*j])**2 + (p[3*i+1] - p[3*j+1])**2 - (p[3*i+2] + p[3*j+2])**2})

        maxiter = int(min(500, time_budget * 100))
        try:
            res = minimize(obj, z, method='SLSQP', bounds=bounds, constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': 1e-12, 'maxls': 50})
            if res.success or res.fun < _objective(z):
                return res.x
        except Exception:
            pass
        return z

    def _generate_initial_guesses():
        guesses = []
        # 1. Grid-based
        side = int(np.ceil(np.sqrt(N)))
        for w0 in [1.0, 0.8, 1.2, 0.7, 1.3, 0.6, 1.4, 0.5, 1.5, 0.4]:
            h0 = 2.0 - w0
            if h0 < 0.3: continue
            z = np.zeros(3 * N + 1)
            for i in range(N):
                row, col = divmod(i, side)
                cx = (col + 0.5) / side * w0
                cy = (row + 0.5) / side * h0
                r0 = min(w0 / side, h0 / side) * 0.45
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
            
        # 2. Delaunay-based (better for non-uniform sizes)
        for w0 in [1.0, 0.9, 1.1, 0.8, 1.2]:
            h0 = 2.0 - w0
            if h0 < 0.3: continue
            # Generate random points
            rng = np.random.default_rng(42)
            pts = rng.uniform([0.1, 0.1], [w0-0.1, h0-0.1], (N, 2))
            try:
                tri = Delaunay(pts)
                # Compute inradii of triangles formed by points and boundaries? 
                # Simpler: Use Voronoi cell inradius approximation or just use Delaunay to ensure connectivity.
                # For simplicity, let's just use the points and set radii based on min distance to neighbors/box.
                z = np.zeros(3 * N + 1)
                for i in range(N):
                    cx, cy = pts[i]
                    # Min distance to other points
                    d_pts = np.linalg.norm(pts - pts[i], axis=1)
                    d_pts[i] = np.inf
                    d_min = np.min(d_pts)
                    d_box = min(cx, w0-cx, cy, h0-cy)
                    r0 = 0.45 * min(d_min, d_box)
                    z[3*i] = cx
                    z[3*i+1] = cy
                    z[3*i+2] = r0
                z[-1] = w0
                guesses.append(z)
            except Exception:
                continue
                
        # 3. Random
        rng = np.random.default_rng(123)
        for _ in range(10):
            w0 = rng.uniform(0.6, 1.4)
            h0 = 2.0 - w0
            z = np.zeros(3 * N + 1)
            for i in range(N):
                r0 = rng.uniform(0.04, 0.14)
                cx = rng.uniform(r0, w0 - r0)
                cy = rng.uniform(r0, h0 - r0)
                z[3*i] = cx
                z[3*i+1] = cy
                z[3*i+2] = r0
            z[-1] = w0
            guesses.append(z)
            
        return guesses

    guesses = _generate_initial_guesses()
    
    for i, z0 in enumerate(guesses):
        if time.time() > deadline - 30:
            break
        
        # Phase 1: Gradient Descent
        z_gd = _gradient_descent(z0, deadline - 20)
        
        # Check validity and value
        x = z_gd[0::3][:N]
        y = z_gd[1::3][:N]
        r = z_gd[2::3][:N]
        w = z_gd[-1]
        h = 2.0 - w
        
        if _is_valid(x, y, r, w, h):
            val = np.sum(r)
            if val > best_value:
                best_value = val
                best_result = z_gd.copy()
                
        # Phase 2: SLSQP Polish
        if time.time() < deadline - 10:
            z_slsqp = _slsqp_polish(z_gd, deadline - 5)
            x = z_slsqp[0::3][:N]
            y = z_slsqp[1::3][:N]
            r = z_slsqp[2::3][:N]
            w = z_slsqp[-1]
            h = 2.0 - w
            
            if _is_valid(x, y, r, w, h):
                val = np.sum(r)
                if val > best_value:
                    best_value = val
                    best_result = z_slsqp.copy()

    if best_result is None:
        # Fallback
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])

    # Final safety shrink
    x = best_result[0::3][:N]
    y = best_result[1::3][:N]
    r = best_result[2::3][:N]
    w = best_result[-1]
    h = 2.0 - w
    
    for _ in range(10):
        r_trial = r * (1 - 1e-9)
        if _is_valid(x, y, r_trial, w, h):
            r = r_trial
        else:
            break

    return np.column_stack((x, y, r))
