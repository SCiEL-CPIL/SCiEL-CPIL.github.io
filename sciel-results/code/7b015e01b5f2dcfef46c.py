import numpy as np
import time
import sys
from scipy.optimize import minimize, basinhopping

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Minimizes the area of the union of 64 triangles using basin hopping
    with Nelder-Mead as the local optimizer.
    """
    n = 64
    # Precompute fixed values
    a = np.arange(1, n + 1) / n  # a_i = (i+1)/64
    # Quadrature heights y_m = (m + 0.5) / 20000
    m = np.arange(20000)
    y = (m + 0.5) / 20000.0
    w = (1.0 - y) / n  # Width at height y, shape (20000,)

    def calculate_area(x):
        """
        x: array of shape (64,)
        Returns: area of the union of triangles
        """
        L = x[None, :] + a[None, :] * y[:, None]
        L_sorted = np.sort(L, axis=1)
        diffs = L_sorted[:, 1:] - L_sorted[:, :-1]
        w_broadcast = w[:, None]
        contribs = np.minimum(diffs, w_broadcast)
        union_lengths = w + np.sum(contribs, axis=1)
        area = np.mean(union_lengths)
        return area

    # Generate initial positions using the seed method (Keich construction)
    def seed_positions():
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)

    x0 = seed_positions()
    
    # Initial evaluation
    best_x = x0.copy()
    best_area = calculate_area(best_x)
    
    start_time = time.time()
    deadline = start_time + 620  # Safety margin for 630s limit
    
    # Define the local optimizer
    def local_minimizer(x):
        options = {'maxiter': 1000, 'maxfev': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
        res = minimize(calculate_area, x, method='Nelder-Mead', options=options)
        return res.x, res.fun, {'nfev': res.nfev}
    
    # Run basin hopping
    try:
        bh = basinhopping(
            calculate_area,
            x0,
            minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 1000, 'maxfev': 5000, 'xatol': 1e-6, 'fatol': 1e-8}},
            niter=50,
            stepsize=0.5,
            T=1.0,
            seed=42
        )
        if bh.fun < best_area:
            best_x = bh.x.copy()
            best_area = bh.fun
    except Exception:
        pass
    
    # Continue with local refinement if time allows
    if time.time() < deadline - 30:
        options = {'maxiter': 5000, 'maxfev': 20000, 'xatol': 1e-8, 'fatol': 1e-10}
        try:
            res = minimize(calculate_area, best_x, method='Nelder-Mead', options=options)
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun
        except Exception:
            pass
    
    return best_x

# EVOLVE-BLOCK-END
