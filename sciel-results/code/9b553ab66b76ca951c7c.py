# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_1d_union_length(x, c, N=8):
    """Compute |∪_{i=0}^{N-1} [x_i + c*i/N, x_i + c*i/N + 1/N]| at a single c"""
    starts = x + c * np.arange(N) / N
    ends = starts + 1.0 / N
    order = np.argsort(starts)
    s_sorted = starts[order]
    e_sorted = ends[order]
    union_len = 0.0
    cur_start = s_sorted[0]
    cur_end = e_sorted[0]
    for i in range(1, N):
        if s_sorted[i] > cur_end:
            union_len += cur_end - cur_start
            cur_start = s_sorted[i]
            cur_end = e_sorted[i]
        else:
            cur_end = max(cur_end, e_sorted[i])
    union_len += cur_end - cur_start
    return union_len

def _compute_3d_volume(x, N=8, n_samples=2000):
    """Compute ∫_0^1 f(c)^2 dc where f(c) = |∪_{i} [x_i + c*i/N, x_i + c*i/N + 1/N]|"""
    c_vals = np.linspace(0, 1, n_samples)
    total = 0.0
    for c in c_vals:
        union_len = _compute_1d_union_length(x, c, N)
        total += union_len * union_len
    return total / n_samples

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 600.0
    
    # Candidate 1: Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    # Candidate 2: Symmetric placement (centered)
    symmetric = -np.arange(N) / (2 * N)  # x_i = -i/16
    
    # Candidate 3: All at origin
    origin = np.zeros(N)
    
    # Candidate 4: Linearly spaced
    linear = np.linspace(0, 1 - 1/N, N)
    
    candidates = [keich, symmetric, origin, linear]
    
    best_x = keich.copy()
    best_vol = float('inf')
    
    for x in candidates:
        vol = _compute_3d_volume(x, N, 1000)
        if vol < best_vol:
            best_vol = vol
            best_x = x.copy()
    
    # Try local optimization around the best candidate
    if time.time() < deadline - 180.0:
        from scipy.optimize import minimize
        
        def objective(x):
            return _compute_3d_volume(x, N, 500)
        
        x0 = best_x.copy()
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 8000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_vol:
                best_vol = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Try L-BFGS-B
        if time.time() < deadline - 120.0:
            try:
                res2 = minimize(objective, best_x, method='L-BFGS-B',
                               options={'maxiter': 800, 'maxls': 50})
                if res2.fun < best_vol:
                    best_vol = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass
    
    # Final high-precision evaluation
    best_vol = _compute_3d_volume(best_x, N, 5000)
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
