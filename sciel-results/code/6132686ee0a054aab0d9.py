# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _calculate_area(x):
    """
    Calculate the area of the union of the triangles for given positions x.
    """
    n = 64
    # Quadrature points
    m = 20000
    y = (np.arange(m) + 0.5) / m
    w = (1 - y) / n
    
    # a_i = (i + 1) / n
    a = (np.arange(n) + 1) / n
    
    # L_i(y) = x_i + a_i * y
    # x shape (n,), a shape (n,), y shape (m,)
    # L shape (n, m)
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L for each y
    L_sorted = np.sort(L, axis=0)
    
    # Calculate gaps between consecutive sorted L values
    gaps = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length at each y: w + sum(min(gaps, w))
    # Note: The first interval contributes w. Subsequent ones contribute min(gap, w).
    # Total = w + sum_{j=1}^{n-1} min(gap_j, w)
    union_lengths = w[np.newaxis, :] + np.sum(np.minimum(gaps, w[np.newaxis, :]), axis=0)
    
    return np.mean(union_lengths)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    
    # Generate Keich positions as initial guess
    keich_positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        pos = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        keich_positions.append(pos)
    x_keich = np.array(keich_positions, dtype=np.float64)
    
    best_x = x_keich
    best_area = _calculate_area(x_keich)
    
    # Try to improve using local optimization
    # Nelder-Mead is robust to non-smoothness
    for _ in range(5):
        res = minimize(_calculate_area, x_keich, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 1000, 'adaptive': True})
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x
            
    # Also try a few random restarts to ensure global improvement
    rng = np.random.default_rng(42)
    for _ in range(10):
        # Random start within a reasonable box
        # Keich positions are in [0, 1) roughly
        x_rand = rng.uniform(0, 1, size=n)
        res = minimize(_calculate_area, x_rand, method='Nelder-Mead', 
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True})
        if res.fun < best_area:
            best_area = res.fun
            best_x = res.x

    return best_x

# EVOLVE-BLOCK-END
