# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def squares_in_square10() -> tuple:
    """
    Maximizes the sum of side lengths of 10 non-overlapping squares in the unit square.
    Uses Differential Evolution with a penalty method.
    """
    n = 10
    start_time = time.time()
    time_limit = 580.0  # 20s safety margin

    # Initial best: Grid packing (Score 1.0)
    k = 3
    side_grid = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx_grid = (cols + 0.5) * side_grid
    cy_grid = (rows + 0.5) * side_grid
    best_x = np.zeros(3 * n)
    best_x[0:3*n:3] = cx_grid
    best_x[1:3*n:3] = cy_grid
    best_x[2:3*n:3] = side_grid
    best_val = np.sum(side_grid)

    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0))
        bounds.append((0.0, 1.0))
        bounds.append((0.01, 1.0))

    def get_params(x):
        return x[0:3*n:3], x[1:3*n:3], x[2:3*n:3]

    def calc_penalty(x):
        cx, cy, s = get_params(x)
        pen = 0.0
        # Boundary
        for i in range(n):
            if cx[i] < s[i]/2:
                pen += (s[i]/2 - cx[i])**2
            if cx[i] > 1.0 - s[i]/2:
                pen += (cx[i] - (1.0 - s[i]/2))**2
            if cy[i] < s[i]/2:
                pen += (s[i]/2 - cy[i])**2
            if cy[i] > 1.0 - s[i]/2:
                pen += (cy[i] - (1.0 - s[i]/2))**2
        
        # Overlap
        for i in range(n):
            for j in range(i+1, n):
                dx = abs(cx[i] - cx[j])
                dy = abs(cy[i] - cy[j])
                S = (s[i] + s[j]) / 2.0
                if dx < S and dy < S:
                    pen += ((S - dx) * (S - dy))**2
        return pen

    def objective(x):
        cx, cy, s = get_params(x)
        pen = calc_penalty(x)
        return -np.sum(s) + 100.0 * pen

    def callback(xk, convergence):
        nonlocal best_x, best_val
        # Check time
        if time.time() - start_time > time_limit:
            return True
        # Check validity
        pen = calc_penalty(xk)
        if pen < 1e-8:
            s = xk[2:3*n:3]
            val = np.sum(s)
            if val > best_val:
                best_val = val
                best_x = xk.copy()
        return False

    try:
        res = scipy.optimize.differential_evolution(
            objective,
            bounds,
            maxiter=150,
            popsize=20,
            tol=1e-8,
            seed=42,
            callback=callback,
            polish=True
        )
        
        # Check if the final result is better
        if res.success:
            pen = calc_penalty(res.x)
            if pen < 1e-8:
                s = res.x[2:3*n:3]
                val = np.sum(s)
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
                    
    except Exception:
        pass

    # Final extraction
    cx = best_x[0:3*n:3]
    cy = best_x[1:3*n:3]
    s = best_x[2:3*n:3]
    
    centers = np.stack([cx, cy], axis=1).astype(np.float64)
    sides = s.astype(np.float64)
    
    return centers, sides

# EVOLVE-BLOCK-END
