import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580
    
    N = 8
    n_points = 5000
    
    def compute_volume(f):
        c_vals = np.linspace(0, 1, n_points)
        i_idx = np.arange(N, dtype=np.float64)
        left = f[None, :] + c_vals[:, None] * i_idx[None, :] / N
        right = left + 1.0 / N
        
        order = np.argsort(left, axis=1)
        sorted_left = np.take_along_axis(left, order, axis=1)
        sorted_right = np.take_along_axis(right, order, axis=1)
        
        union_len = np.empty(n_points)
        for k in range(n_points):
            cur_start = sorted_left[k, 0]
            cur_end = sorted_right[k, 0]
            total = 0.0
            for m in range(1, N):
                if sorted_left[k, m] <= cur_end:
                    if sorted_right[k, m] > cur_end:
                        cur_end = sorted_right[k, m]
                else:
                    total += cur_end - cur_start
                    cur_start = sorted_left[k, m]
                    cur_end = sorted_right[k, m]
            total += cur_end - cur_start
            union_len[k] = total
        
        V = np.trapz(union_len ** 2, c_vals)
        return V
    
    # Initial guess: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    best_f = keich.copy()
    best_V = compute_volume(best_f)
    
    def timeout_callback(xk):
        if time.time() > deadline:
            raise Exception("timeout")
        return False
    
    # Main optimization
    try:
        result = minimize(compute_volume, best_f, method='Nelder-Mead',
                          options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12},
                          callback=timeout_callback)
        if result.fun < best_V:
            best_V = result.fun
            best_f = result.x.copy()
    except Exception:
        pass
    
    # Try random restarts if time permits
    rng = np.random.default_rng(42)
    while time.time() < deadline - 30:
        f0 = rng.uniform(-1, 1, N)
        try:
            result = minimize(compute_volume, f0, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12},
                              callback=timeout_callback)
            if result.fun < best_V:
                best_V = result.fun
                best_f = result.x.copy()
        except Exception:
            break
    
    # Construct positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_f[i_flat]
    base_y = best_f[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    return positions
