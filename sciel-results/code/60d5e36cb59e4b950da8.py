# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _keich_positions(n=64):
    """Classical Keich construction for n a power of 2."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
    return np.array(positions, dtype=np.float64)


def _compute_area_and_grad(x):
    """
    Compute the union area and its analytic gradient.
    
    The area is computed via quadrature at M = 20000 heights.
    At each height y, triangle i has cross-section [L_i, L_i + w_i] where
    L_i = x_i + a_i * y and w_i = (1 - y)/n.
    The union length is w + sum of min(gaps, w) over sorted consecutive pairs.
    """
    n = 64
    M = 20000
    a = np.arange(1, n + 1, dtype=np.float64) / n
    y = (np.arange(M) + 0.5) / M
    w = (1.0 - y) / n  # width at each height, shape (M,)

    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[None, :] + a[None, :] * y[:, None]

    # Sort L along axis 1 (for each height)
    sort_idx = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, sort_idx, axis=1)

    # Consecutive differences, shape (M, n-1)
    d = L_sorted[:, 1:] - L_sorted[:, :-1]

    # Union length at each height
    w_col = w[:, None]
    d_clipped = np.minimum(d, w_col)
    U = w + np.sum(d_clipped, axis=1)  # shape (M,)

    area = np.mean(U)

    # Gradient computation
    # For each height m and consecutive pair j, if d[m,j] < w[m],
    # then the term min(d, w) = d contributes to the gradient.
    # d[m,j] = L_sorted[m, j+1] - L_sorted[m, j]
    # So dL_sorted[m, j+1]/dx contributes +1 and dL_sorted[m, j]/dx contributes -1
    # L_sorted[m, j] = L[m, sort_idx[m, j]] = x[sort_idx[m, j]] + a[sort_idx[m, j]] * y[m]
    # So dL_sorted[m, j]/dx_i = 1 if sort_idx[m, j] == i, else 0

    mask = d < w_col  # shape (M, n-1)
    m_indices, j_indices = np.where(mask)

    grad_L = np.zeros_like(L)  # shape (M, n)
    # For each (m, j) in mask:
    #   grad_L[m, sort_idx[m, j+1]] += 1
    #   grad_L[m, sort_idx[m, j]] -= 1
    if len(m_indices) > 0:
        np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices + 1]), 1.0)
        np.add.at(grad_L, (m_indices, sort_idx[m_indices, j_indices]), -1.0)

    # grad_x[i] = sum_m grad_L[m, i] / M
    grad_x = np.sum(grad_L, axis=0) / M

    return area, grad_x


def _compute_area_only(x):
    """Compute only the union area (no gradient)."""
    n = 64
    M = 20000
    a = np.arange(1, n + 1, dtype=np.float64) / n
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
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Uses Keich construction as starting point, then optimizes with L-BFGS-B
    using analytic gradients, followed by multi-restart perturbation and final polish.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit

    # Initial guess: Keich construction
    x0 = _keich_positions(n)

    # Phase 1: L-BFGS-B with analytic gradient from Keich start
    best_x = x0.copy()
    best_area = _compute_area_only(best_x)

    try:
        result = minimize(
            lambda x: _compute_area_and_grad(x)[0],
            x0,
            jac=lambda x: _compute_area_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 5000, 'maxfun': 50000, 'disp': False, 'ftol': 1e-15, 'gtol': 1e-10}
        )
        if result.fun < best_area:
            best_x = result.x.copy()
            best_area = result.fun
    except Exception:
        pass

    # Phase 2: Multiple restarts with perturbation to escape local optima
    rng = np.random.default_rng(42)
    for restart in range(8):
        if time.time() > deadline - 120:
            break

        # Perturb the current best with decreasing scale
        scale = 0.02 / (restart + 1)
        x_pert = best_x + rng.normal(0, scale, size=n)

        try:
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
        except Exception:
            pass

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 30:
        try:
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
        except Exception:
            pass

    # Ensure validity
    if not np.all(np.isfinite(best_x)):
        best_x = x0.copy()

    return best_x.astype(np.float64)


# EVOLVE-BLOCK-END
