# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_area_exact(x, n=64, n_heights=20000):
    """Compute the area of the union of triangles for given base positions x (full precision)."""
    a = (np.arange(n) + 1) / n
    y_vals = (np.arange(n_heights) + 0.5) / n_heights
    w_vals = (1.0 - y_vals) / n
    L = x[None, :] + a[None, :] * y_vals[:, None]
    L_sorted = np.sort(L, axis=1)
    gaps = np.diff(L_sorted, axis=1)
    union_lens = w_vals + np.sum(np.minimum(gaps, w_vals[:, None]), axis=1)
    return np.mean(union_lens)

def _union_area_and_grad(x, M=4000):
    """
    Compute the union area of 64 triangles and its gradient w.r.t. x.
    Uses the analytical gradient via bincount for speed.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    mask = diffs < w[:, np.newaxis]
    
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    m_idx, j_idx = np.where(mask)
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        grad_right = np.bincount(right_indices, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, minlength=n).astype(np.float64)
        grad = grad_right - grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M
    
    return area, grad

def _softmin_objective(x, beta=50.0, M=4000):
    """
    Smooth surrogate for the union area using soft-min approximation.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    L_sorted = np.sort(L, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    neg_beta_diffs = -beta * diffs
    neg_beta_w = -beta * w[:, np.newaxis]
    
    max_vals = np.maximum(neg_beta_diffs, neg_beta_w)
    log_sum_exp = max_vals + np.log(np.exp(neg_beta_diffs - max_vals) + np.exp(neg_beta_w - max_vals))
    
    soft_min = -log_sum_exp / beta
    
    union_len = w + np.sum(soft_min, axis=1)
    return np.mean(union_len)

def _softmin_gradient(x, beta=50.0, M=4000):
    """
    Compute gradient of the soft-min surrogate objective.
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    neg_beta_diffs = -beta * diffs
    neg_beta_w = -beta * w[:, np.newaxis]
    
    max_vals = np.maximum(neg_beta_diffs, neg_beta_w)
    exp_diffs = np.exp(neg_beta_diffs - max_vals)
    exp_w = np.exp(neg_beta_w - max_vals)
    
    denom = exp_diffs + exp_w
    # Avoid division by zero
    denom = np.where(denom == 0, 1e-300, denom)
    
    weight_right = -exp_diffs / denom
    weight_left = exp_diffs / denom
    
    m_idx, j_idx = np.where(np.ones_like(diffs, dtype=bool))
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        grad_left = np.bincount(left_indices, weights=weight_left[m_idx, j_idx], minlength=n)
        grad_right = np.bincount(right_indices, weights=weight_right[m_idx, j_idx], minlength=n)
        grad = grad_right + grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M
    
    return grad

def _keich_initial(n=64):
    """Generate the Keich construction starting point."""
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    x0 = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
    return x0

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    start_time = time.time()
    deadline = start_time + 600.0  # Safety margin before 630s kill
    
    # Start from the Keich construction
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area_exact(x0)
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: L-BFGS-B with analytical gradient from Keich point
    if time.time() < deadline - 100:
        result = minimize(
            lambda x: _union_area_and_grad(x, M=4000)[0],
            x0,
            method='L-BFGS-B',
            jac=lambda x: _union_area_and_grad(x, M=4000)[1],
            options={
                'maxiter': 300,
                'maxfun': 600,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 30,
            }
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    # Strategy 2: Soft-min L-BFGS-B from best point (smooths the nonsmooth min)
    if time.time() < deadline - 80:
        result = minimize(
            lambda x: _softmin_objective(x, beta=50.0, M=4000),
            best_x,
            method='L-BFGS-B',
            jac=lambda x: _softmin_gradient(x, beta=50.0, M=4000),
            options={
                'maxiter': 300,
                'maxfun': 600,
                'ftol': 1e-14,
                'gtol': 1e-10,
                'maxls': 30,
            }
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    # Strategy 3: Nelder-Mead restarts with varied perturbations
    for restart in range(10):
        if time.time() > deadline - 40:
            break
        scale = 0.002 * (1 + restart * 0.4)
        x_start = best_x + rng.normal(0, scale, size=n)
        result = minimize(
            lambda x: _union_area_and_grad(x, M=3000)[0],
            x_start,
            method='Nelder-Mead',
            options={'maxiter': 8000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
    
    # Strategy 4: Try a few random starts with L-BFGS-B
    if time.time() < deadline - 60:
        for trial in range(4):
            if time.time() > deadline - 40:
                break
            x_rand = rng.uniform(-0.5, 0.5, size=n)
            result = minimize(
                lambda x: _union_area_and_grad(x, M=4000)[0],
                x_rand,
                method='L-BFGS-B',
                jac=lambda x: _union_area_and_grad(x, M=4000)[1],
                options={
                    'maxiter': 200,
                    'maxfun': 400,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20,
                }
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
    
    # Strategy 5: Final polish with high-precision soft-min then Nelder-Mead
    if time.time() < deadline - 30:
        # Soft-min polish
        result = minimize(
            lambda x: _softmin_objective(x, beta=100.0, M=5000),
            best_x,
            method='L-BFGS-B',
            jac=lambda x: _softmin_gradient(x, beta=100.0, M=5000),
            options={
                'maxiter': 200,
                'maxfun': 400,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 20,
            }
        )
        if result.fun < best_area:
            best_area = result.fun
            best_x = result.x.copy()
        
        # Nelder-Mead final refinement
        if time.time() < deadline - 15:
            result = minimize(
                lambda x: _union_area_and_grad(x, M=5000)[0],
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-13, 'fatol': 1e-15, 'adaptive': True}
            )
            if result.fun < best_area:
                best_area = result.fun
                best_x = result.x.copy()
    
    # Verify with full precision
    final_area = _compute_area_exact(best_x)
    if final_area <= 0 or not np.all(np.isfinite(best_x)):
        best_x = x0
    
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
