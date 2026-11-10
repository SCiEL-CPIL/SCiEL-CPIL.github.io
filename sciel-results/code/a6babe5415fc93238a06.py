# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _compute_area_and_grad(x):
    n = 64
    M = 20000
    a = np.arange(1, n + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n

    # L shape (M, n)
    L = x[None, :] + a[None, :] * y[:, None]

    # Sort L along axis 1
    sort_idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, sort_idx, axis=1)

    # d shape (M, n-1)
    d = L_sorted[:, 1:] - L_sorted[:, :-1]

    # Union length U shape (M,)
    w_col = w[:, None]
    d_clipped = np.minimum(d, w_col)
    U = w + np.sum(d_clipped, axis=1)

    area = np.mean(U)

    # Gradient
    mask = d < w_col
    m_indices = np.where(mask)[0]
    j_indices = np.where(mask)[1]

    grad_L = np.zeros_like(L)
    np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices]), -1.0)
    np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices + 1]), 1.0)

    grad_x = np.sum(grad_L, axis=0) / M

    return area, grad_x

def _compute_area_only(x):
    n = 64
    M = 20000
    a = np.arange(1, n + 1) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n

    L = x[None, :] + a[None, :] * y[:, None]
    sort_idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, sort_idx, axis=1)
    d = L_sorted[:, 1:] - L_sorted[:, :-1]
    w_col = w[:, None]
    d_clipped = np.minimum(d, w_col)
    U = w + np.sum(d_clipped, axis=1)

    return np.mean(U)

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    x0 = np.array(positions, dtype=np.float64)

    start_time = time.time()
    deadline = start_time + 590

    # Phase 1: L-BFGS-B with analytic gradient
    result = minimize(
        lambda x: _compute_area_and_grad(x)[0],
        x0,
        jac=lambda x: _compute_area_and_grad(x)[1],
        method='L-BFGS-B',
        options={'maxiter': 5000, 'maxfun': 50000, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-10}
    )

    best_x = result.x.copy()
    best_area = _compute_area_only(best_x)

    # Phase 2: Multiple restarts with perturbation to escape local optima
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 120:
            break
        # Perturb the current best
        scale = 0.01 / (restart + 1)
        x_pert = best_x + rng.normal(0, scale, size=n)
        
        res = minimize(
            lambda x: _compute_area_and_grad(x)[0],
            x_pert,
            jac=lambda x: _compute_area_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 3000, 'maxfun': 30000, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        
        if res.fun < best_area:
            best_x = res.x.copy()
            best_area = res.fun

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 30:
        res_final = minimize(
            lambda x: _compute_area_and_grad(x)[0],
            best_x,
            jac=lambda x: _compute_area_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 10000, 'maxfun': 100000, 'disp': False, 'ftol': 1e-16, 'gtol': 1e-12}
        )
        if res_final.fun < best_area:
            best_x = res_final.x.copy()
            best_area = res_final.fun

    # Ensure validity
    if not np.all(np.isfinite(best_x)):
        best_x = x0.copy()

    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
