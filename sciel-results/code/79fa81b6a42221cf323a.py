# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0
    
    def objective(params):
        # params: [cx_0, cy_0, s_0, cx_1, cy_1, s_1, ...]
        # Minimize negative sum of sides
        s = params[2::3]
        return -np.sum(s)
    
    def constraints(params):
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        cons = []
        # Boundary constraints: each square inside [0,1]^2
        for i in range(n):
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i] - p[3*i+2]/2 - 0.0})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i] - p[3*i+2]/2})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+1] - p[3*i+2]/2 - 0.0})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: 1.0 - p[3*i+1] - p[3*i+2]/2})
            cons.append({'type': 'ineq', 'fun': lambda p, i=i: p[3*i+2] - 1e-10})  # side > 0
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                # |cx_i - cx_j| >= t OR |cy_i - cy_j| >= t
                # We need to handle this as: min(|dx|-t, |dy|-t) >= 0 is NOT correct
                # Actually overlap if |dx| < t AND |dy| < t
                # Non-overlap: |dx| >= t OR |dy| >= t
                # This is a disjunctive constraint. For SLSQP, we can't do OR directly.
                # Alternative: use a smooth penalty or reformulate.
                # Let's use: (|dx| - t) * (|dy| - t) >= 0 doesn't work either.
                # Better: penalize overlap in objective, or use different approach.
                pass
        return cons
    
    # Since disjunctive constraints are hard for SLSQP, use penalty approach
    def penalized_objective(params, penalty):
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        obj = -np.sum(s)
        # Boundary penalties
        for i in range(n):
            if cx[i] - s[i]/2 < 0:
                obj += penalty * (0 - (cx[i] - s[i]/2))**2
            if cx[i] + s[i]/2 > 1:
                obj += penalty * ((cx[i] + s[i]/2) - 1)**2
            if cy[i] - s[i]/2 < 0:
                obj += penalty * (0 - (cy[i] - s[i]/2))**2
            if cy[i] + s[i]/2 > 1:
                obj += penalty * ((cy[i] + s[i]/2) - 1)**2
            if s[i] < 1e-10:
                obj += penalty * (1e-10 - s[i])**2
        # Overlap penalties
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                # Overlap if dx < t and dy < t
                # Penalty: max(0, t - dx) * max(0, t - dy)
                ox = max(0, t - dx)
                oy = max(0, t - dy)
                obj += penalty * ox * oy
        return obj
    
    def is_valid(params):
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        if not np.all(np.isfinite(params)):
            return False
        if np.any(s <= 0):
            return False
        for i in range(n):
            if cx[i] - s[i]/2 < -tol:
                return False
            if cx[i] + s[i]/2 > 1 + tol:
                return False
            if cy[i] - s[i]/2 < -tol:
                return False
            if cy[i] + s[i]/2 > 1 + tol:
                return False
        for i in range(n):
            for j in range(i+1, n):
                t = (s[i] + s[j]) / 2.0 - tol
                if abs(cx[i] - cx[j]) < t and abs(cy[i] - cy[j]) < t:
                    return False
        return True
    
    def get_score(params):
        if not is_valid(params):
            return -np.inf
        s = params[2::3]
        return np.sum(s)
    
    # Initial guesses
    def grid_init(k):
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        s = np.full(n, side)
        params = np.zeros(3*n)
        params[0::3] = cx
        params[1::3] = cy
        params[2::3] = s
        return params
    
    def asymmetric_init():
        # One large square in corner, others around
        s0 = 0.5
        cx0, cy0 = 0.25, 0.25
        # Remaining space: right strip [0.5,1]x[0,1], top strip [0,1]x[0.5,1]
        # Place 9 squares in the L-shaped region
        # Right strip: width 0.5, height 1.0 -> can fit squares of side 0.5
        # Top strip: width 1.0, height 0.5 -> can fit squares of side 0.5
        # But they overlap in [0.5,1]x[0.5,1]
        # Let's try: 3 squares of side 0.25 in right strip, 3 in top strip, 3 in corner
        cx = np.zeros(n)
        cy = np.zeros(n)
        s = np.zeros(n)
        cx[0], cy[0], s[0] = 0.25, 0.25, 0.5
        # Right strip: x in [0.5, 1], y in [0, 1]
        for i in range(3):
            cx[1+i] = 0.75
            cy[1+i] = (i + 0.5) * (1.0/3)
            s[1+i] = 1.0/3
        # Top strip: x in [0, 0.5], y in [0.5, 1]
        for i in range(3):
            cx[4+i] = (i + 0.5) * (0.5/3)
            cy[4+i] = 0.75
            s[4+i] = 1.0/3
        # Corner [0.5,1]x[0.5,1]: 3 squares of side 1/6
        for i in range(3):
            cx[7+i] = 0.5 + (i + 0.5) * (0.5/3)
            cy[7+i] = 0.75
            s[7+i] = 1.0/3
        params = np.zeros(3*n)
        params[0::3] = cx
        params[1::3] = cy
        params[2::3] = s
        return params
    
    def random_init(rng):
        cx = rng.uniform(0.1, 0.9, n)
        cy = rng.uniform(0.1, 0.9, n)
        s = rng.uniform(0.1, 0.3, n)
        params = np.zeros(3*n)
        params[0::3] = cx
        params[1::3] = cy
        params[2::3] = s
        return params
    
    # Try multiple initializations and optimize
    best_score = -np.inf
    best_params = None
    
    inits = [
        grid_init(4),  # 4x4 grid, 10 squares
        grid_init(3),  # 3x3 grid (only 9, need adjustment)
        asymmetric_init(),
    ]
    
    rng = np.random.default_rng(42)
    for _ in range(20):
        inits.append(random_init(rng))
    
    penalty = 100.0
    
    for init_idx, init_params in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # SLSQP with penalty
        bounds = []
        for i in range(n):
            bounds.append((0, 1))  # cx
            bounds.append((0, 1))  # cy
            bounds.append((1e-10, 1.0))  # s
        
        try:
            result = minimize(
                lambda p: penalized_objective(p, penalty),
                init_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if result.success or is_valid(result.x):
                score = get_score(result.x)
                if score > best_score:
                    best_score = score
                    best_params = result.x.copy()
        except Exception:
            pass
        
        # Also try with higher penalty
        penalty2 = 1000.0
        try:
            result2 = minimize(
                lambda p: penalized_objective(p, penalty2),
                init_params,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            if is_valid(result2.x):
                score2 = get_score(result2.x)
                if score2 > best_score:
                    best_score = score2
                    best_params = result2.x.copy()
        except Exception:
            pass
    
    # Fallback: use the grid solution if no valid result found
    if best_params is None or not is_valid(best_params):
        k = int(np.ceil(np.sqrt(n)))
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides
    
    cx = best_params[0::3]
    cy = best_params[1::3]
    s = best_params[2::3]
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    return centers, sides

# EVOLVE-BLOCK-END
