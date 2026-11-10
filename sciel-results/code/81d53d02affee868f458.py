# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_1d_union_volume_vec(x, N=8, n_samples=2000):
    """Vectorized computation of ∫_0^1 |∪_{i=0}^{N-1} [x_i + c*i/N, x_i + c*i/N + 1/N]| dc"""
    c_vals = np.linspace(0, 1, n_samples)
    # Shape: (n_samples, N)
    starts = x[np.newaxis, :] + np.outer(c_vals, np.arange(N) / N)
    ends = starts + 1.0 / N
    order = np.argsort(starts, axis=1)
    s_sorted = np.take_along_axis(starts, order, axis=1)
    e_sorted = np.take_along_axis(ends, order, axis=1)
    
    cur_start = s_sorted[:, 0:1]
    cur_end = e_sorted[:, 0:1]
    union_len = np.zeros(n_samples)
    for i in range(1, N):
        s_i = s_sorted[:, i:i+1]
        e_i = e_sorted[:, i:i+1]
        overlap = s_i <= cur_end
        cur_end = np.where(overlap, np.maximum(cur_end, e_i), e_i)
        cur_start = np.where(overlap, cur_start, s_i)
        union_len = np.where(overlap, union_len, union_len + (cur_end - cur_start))
    union_len = union_len + (cur_end - cur_start)
    return np.mean(union_len)

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    deadline = time.time() + 570.0
    
    def proxy_volume(x):
        """Proxy for 3D volume: (span + 1/8)^2 * U(x) / 8 where U is 1D union integral."""
        span = np.max(x) - np.min(x)
        box_len = span + 1.0/N
        u = _compute_1d_union_volume_vec(x, N, 400)
        return box_len * box_len * u / N
    
    rng = np.random.default_rng(42)
    
    # Keich construction
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
    
    symmetric = -np.arange(N) / (2 * N)
    origin = np.zeros(N)
    linear = np.linspace(0, 1 - 1/N, N)
    
    candidates = [keich, symmetric, origin, linear]
    for _ in range(8):
        candidates.append(rng.uniform(-0.4, 0.4, N))
    
    best_x = keich.copy()
    best_vol = float('inf')
    
    for x in candidates:
        vol = proxy_volume(x)
        if vol < best_vol:
            best_vol = vol
            best_x = x.copy()
    
    from scipy.optimize import minimize
    
    # Nelder-Mead optimization
    if time.time() < deadline - 150.0:
        try:
            res = minimize(proxy_volume, best_x.copy(), method='Nelder-Mead',
                          options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-10,
                                   'adaptive': True})
            if res.fun < best_vol:
                best_vol = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
    
    # L-BFGS-B
    if time.time() < deadline - 100.0:
        try:
            res2 = minimize(proxy_volume, best_x.copy(), method='L-BFGS-B',
                           options={'maxiter': 400, 'maxls': 50, 'ftol': 1e-14})
            if res2.fun < best_vol:
                best_vol = res2.fun
                best_x = res2.x.copy()
        except Exception:
            pass
    
    # Polish with more samples
    if time.time() < deadline - 50.0:
        def objective_polish(x):
            span = np.max(x) - np.min(x)
            box_len = span + 1.0/N
            u = _compute_1d_union_volume_vec(x, N, 1500)
            return box_len * box_len * u / N
        
        try:
            res3 = minimize(objective_polish, best_x.copy(), method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'adaptive': True})
            if res3.fun < best_vol:
                best_vol = res3.fun
                best_x = res3.x.copy()
        except Exception:
            pass
    
    # More random restarts if time permits
    if time.time() < deadline - 20.0:
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            r = rng.uniform(-0.3, 0.3, N)
            try:
                res_r = minimize(proxy_volume, r, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-10,
                                        'adaptive': True})
                if res_r.fun < best_vol:
                    best_vol = res_r.fun
                    best_x = res_r.copy()
            except Exception:
                pass
    
    # Build the 64 positions
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x = best_x[i_flat]
    base_y = best_x[j_flat]
    positions = np.stack([base_x, base_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
