import numpy as np
import time
from scipy.optimize import linprog

def ring_routing() -> np.ndarray:
    n = 10
    D = 15
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2),
    ]

    # Precompute edge usage matrices
    cw_mat = np.zeros((D, n), dtype=np.float64)
    ccw_mat = np.zeros((D, n), dtype=np.float64)
    weights = np.zeros(D, dtype=np.float64)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        d_cw = (t - s) % n
        if d_cw == 0:
            pass
        else:
            for k in range(d_cw):
                e = (s + k) % n
                cw_mat[i, e] = 1.0
            for e in range(n):
                if cw_mat[i, e] == 0.0:
                    ccw_mat[i, e] = 1.0

    # LP Formulation for initial solution
    c = np.zeros(D + 1)
    c[D] = 1.0
    
    A_ub = np.zeros((n, D + 1))
    b_ub = np.zeros(n)
    
    base_load = weights @ cw_mat
    
    for e in range(n):
        diff = (ccw_mat[:, e] - cw_mat[:, e]) * weights
        A_ub[e, :D] = diff
        A_ub[e, D] = -1.0
        b_ub[e] = -base_load[e]
    
    bounds = [(0, 1)] * D + [(0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            x_lp = res.x[:D]
            x_int = np.round(x_lp).astype(np.int8)
            x_int[x_int < 0.5] = 0
            x_int[x_int >= 0.5] = 1
        else:
            x_int = np.random.randint(0, 2, D).astype(np.int8)
    except Exception:
        x_int = np.random.randint(0, 2, D).astype(np.int8)
        
    def get_max_load(x):
        # x is 0/1 array
        if np.any(x == 1):
            delta_sum = (ccw_mat[x == 1] * weights[x == 1, None]).sum(axis=0)
        else:
            delta_sum = np.zeros(n)
        load = base_load + delta_sum
        return np.max(load)
    
    best_x = x_int.copy()
    best_val = get_max_load(best_x)
    
    # Local Search to refine
    improved = True
    while improved:
        improved = False
        for i in range(D):
            trial_x = best_x.copy()
            trial_x[i] = 1 - trial_x[i]
            val = get_max_load(trial_x)
            if val < best_val - 1e-9:
                best_val = val
                best_x = trial_x
                improved = True
        
        if not improved:
            for i in range(D):
                for j in range(i + 1, D):
                    trial_x = best_x.copy()
                    trial_x[i] = 1 - trial_x[i]
                    trial_x[j] = 1 - trial_x[j]
                    val = get_max_load(trial_x)
                    if val < best_val - 1e-9:
                        best_val = val
                        best_x = trial_x
                        improved = True
                        break
                if improved:
                    break

    # Brute force all 2^15 combinations to guarantee global optimum
    # This is fast enough (32768 * 15 operations)
    delta_mat = (ccw_mat - cw_mat) * weights[:, None]
    
    best_val_brute = np.inf
    best_x_brute = best_x.copy()
    
    # Use a more efficient brute force with vectorized updates where possible
    # But simple loop is fine for 32k
    for mask in range(1 << D):
        load = base_load.copy()
        for i in range(D):
            if mask & (1 << i):
                load += delta_mat[i]
        
        max_load = np.max(load)
        if max_load < best_val_brute - 1e-9:
            best_val_brute = max_load
            best_x_brute = np.array([(mask >> i) & 1 for i in range(D)], dtype=np.int8)

    out = best_x_brute.astype(np.float64)
    return out

ring_routing()
