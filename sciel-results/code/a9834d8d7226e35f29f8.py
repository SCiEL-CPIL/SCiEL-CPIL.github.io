import numpy as np
from scipy.optimize import differential_evolution

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    a = np.arange(1, n + 1) / n
    y = (np.arange(20000) + 0.5) / 20000.0
    w = (1 - y) / n

    def area(x):
        # x is shape (n,)
        # L_i = x_i + a_i * y
        # L shape (n, len(y))
        L = x[:, None] + a[:, None] * y[None, :]
        L_sorted = np.sort(L, axis=0)
        diffs = np.diff(L_sorted, axis=0)
        # union length = w + sum(min(diffs, w))
        union_len = w[None, :] + np.sum(np.minimum(diffs, w[None, :]), axis=0)
        return np.mean(union_len)

    # Bounds for x_i
    # The triangles can shift, but relative positions matter.
    # A range of [-1, 1] is safe.
    bounds = [(-1.0, 1.0) for _ in range(n)]
    
    # Use Differential Evolution for global search
    res = differential_evolution(
        area, 
        bounds, 
        seed=42, 
        maxiter=200, 
        popsize=15, 
        tol=1e-8, 
        polish=True,
        init='latinhypercube'
    )
    
    return res.x
