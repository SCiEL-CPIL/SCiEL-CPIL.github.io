import numpy as np
from scipy.optimize import minimize

def compute_area(x):
    n = 64
    a = (np.arange(n) + 1) / n
    m = 20000
    y = (np.arange(m) + 0.5) / m
    L = x[:, None] + a[None, :] * y[None, :]
    L_sorted = np.sort(L, axis=0)
    w = (1 - y) / n
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    min_gaps = np.minimum(gaps, w[None, :])
    union_width = w + np.sum(min_gaps, axis=0)
    return np.mean(union_width)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    x0 = np.zeros(n)
    
    # Optimize from 0
    res = minimize(compute_area, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
    
    # Optimize from seed
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    seed_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        seed_positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x_seed = np.array(seed_positions, dtype=np.float64)
    
    res2 = minimize(compute_area, x_seed, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-5})
    
    best_x = x0
    best_area = compute_area(x0)
    
    if res.fun < best_area:
        best_x = res.x
        best_area = res.fun
        
    if res2.fun < best_area:
        best_x = res2.x
        best_area = res2.fun
        
    return best_x
