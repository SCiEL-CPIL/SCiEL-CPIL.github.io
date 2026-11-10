# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    w = 1.0 / n
    a = np.arange(1, n + 1) / n
    M = 20000
    y = (np.arange(M) + 0.5) / M
    
    deadline = time.time() + 570
    
    # Keich construction
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    
    def _calc_area(x):
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        gaps = L_sorted[:, 1:] - L_sorted[:, :-1]
        union_lengths = w + np.sum(np.minimum(gaps, w), axis=1)
        return np.mean(union_lengths)
    
    best_x = x0.copy()
    best_area = _calc_area(best_x)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            _calc_area,
            best_x,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_area:
            best_x = result.x
            best_area = result.fun
    except Exception:
        pass
    
    if time.time() > deadline:
        return np.asarray(best_x, dtype=np.float64)
    
    # L-BFGS-B with numerical gradient for further refinement
    try:
        result2 = minimize(
            _calc_area,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        if result2.fun < best_area:
            best_x = result2.x
            best_area = result2.fun
    except Exception:
        pass
    
    best_x = np.asarray(best_x, dtype=np.float64)
    if not np.all(np.isfinite(best_x)):
        return x0
    
    return best_x

# EVOLVE-BLOCK-END
