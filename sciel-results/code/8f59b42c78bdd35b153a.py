# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _union_area_and_grad(x, M=4000):
    """
    Compute the union area of 64 triangles and its gradient w.r.t. x.
    
    x: shape (64,) - base positions
    M: number of quadrature points
    Returns: (area, grad) where grad has shape (64,)
    """
    n = 64
    a = (np.arange(n) + 1) / n  # a_i = (i+1)/64
    y = (np.arange(M) + 0.5) / M  # shape (M,)
    
    # L_i(y) = x_i + a_i * y, shape (M, n)
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n  # shape (M,)
    
    # Sort L along axis=1 and track permutation
    order = np.argsort(L, axis=1)  # shape (M, n)
    L_sorted = np.take_along_axis(L, order, axis=1)  # shape (M, n)
    
    # Consecutive differences
    diffs = np.diff(L_sorted, axis=1)  # shape (M, n-1)
    
    # mask: (M, n-1) boolean, True where diffs < w
    mask = diffs < w[:, np.newaxis]  # shape (M, n-1)
    
    # union_len at each y
    union_len = w + np.sum(np.minimum(diffs, w[:, np.newaxis]), axis=1)
    area = np.mean(union_len)
    
    # Gradient computation using bincount for speed
    m_idx, j_idx = np.where(mask)
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        grad_right = np.bincount(right_indices, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, minlength=n).astype(np.float64)
        grad = grad_right - grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M  # mean over quadrature points
    
    return area, grad


def _softmin_area_and_grad(x, beta=50.0, M=4000):
    """
    Smooth surrogate for the union area using soft-min approximation.
    Returns (area, grad) where grad has shape (64,).
    """
    n = 64
    a = (np.arange(n) + 1) / n
    y = (np.arange(M) + 0.5) / M
    
    L = x[np.newaxis, :] + a[np.newaxis, :] * y[:, np.newaxis]
    w = (1.0 - y) / n
    
    order = np.argsort(L, axis=1)
    L_sorted = np.take_along_axis(L, order, axis=1)
    diffs = np.diff(L_sorted, axis=1)
    
    # soft_min(d, w) = -1/beta * log(exp(-beta*d) + exp(-beta*w))
    # Using log-sum-exp trick for stability
    neg_beta_diffs = -beta * diffs
    neg_beta_w = -beta * w[:, np.newaxis]
    
    max_vals = np.maximum(neg_beta_diffs, neg_beta_w)
    exp_diffs = np.exp(neg_beta_diffs - max_vals)
    exp_w = np.exp(neg_beta_w - max_vals)
    log_sum_exp = max_vals + np.log(exp_diffs + exp_w)
    
    soft_min = -log_sum_exp / beta
    union_len = w + np.sum(soft_min, axis=1)
    area = np.mean(union_len)
    
    # Gradient: d(soft_min)/d(diff) = -exp_diffs / (exp_diffs + exp_w)
    # d(soft_min)/d(L_sorted[j+1]) = -exp_diffs/(exp_diffs+exp_w)
    # d(soft_min)/d(L_sorted[j]) = exp_diffs/(exp_diffs+exp_w)
    
    denom = exp_diffs + exp_w
    weight_right = -exp_diffs / denom  # d(soft_min)/d(L_sorted[j+1])
    weight_left = exp_diffs / denom    # d(soft_min)/d(L_sorted[j])
    
    # Accumulate gradients for all pairs
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
    
    return area, grad


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


def _structured_perturbation(x, rng, scale=0.1):
    """
    Apply structured perturbations: subset shifts, pairwise swaps.
    Returns a perturbed copy.
    """
    n = len(x)
    x_new = x.copy()
    
    # Random subset shift
    subset_size = rng.integers(4, 16)
    subset = rng.choice(n, size=subset_size, replace=False)
    x_new[subset] += rng.normal(0, scale, size=subset_size)
    
    # Pairwise swap perturbation
    if rng.random() < 0.5:
        i, j = rng.choice(n, size=2, replace=False)
        swap_amount = rng.normal(0, scale * 0.5)
        x_new[i] += swap_amount
        x_new[j] -= swap_amount
    
    return x_new


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses the Keich construction as initialization, then refines with:
    1. L-BFGS-B with analytical gradient at reduced resolution (M=4000)
    2. Soft-min surrogate optimization with adaptive beta for smooth navigation
    3. Nelder-Mead polish at full resolution (M=20000) for final refinement
    Multiple restarts with structured perturbations to escape local minima.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    start_time = time.time()
    deadline = start_time + 630.0
    safety_margin = 30.0
    
    # Keich construction as initial guess
    x0 = _keich_initial(n)
    best_x = x0.copy()
    best_area = _compute_area_exact(best_x)
    
    rng = np.random.default_rng(42)
    
    # Generate initial guesses: Keich + structured perturbations + random
    initial_guesses = [x0.copy()]
    
    # Structured perturbations at various scales
    for attempt in range(10):
        scale = 0.3 / (attempt + 1)
        perturbed = _structured_perturbation(x0, rng, scale)
        initial_guesses.append(perturbed)
    
    # Gaussian perturbations
    for attempt in range(5):
        perturbation_scale = 0.5 / (attempt + 1)
        perturbed = x0 + rng.normal(0, perturbation_scale, size=n)
        initial_guesses.append(perturbed)
    
    # Fully random starts
    for attempt in range(5):
        random_start = rng.uniform(-1, 1, size=n)
        initial_guesses.append(random_start)
    
    # Phase 1: L-BFGS-B with analytical gradient (fast M=4000)
    for idx, x_start in enumerate(initial_guesses):
        if time.time() > deadline - safety_margin - 120.0:
            break
        
        try:
            result = minimize(
                lambda x: _union_area_and_grad(x, M=4000)[0],
                x_start,
                method='L-BFGS-B',
                jac=lambda x: _union_area_and_grad(x, M=4000)[1],
                options={
                    'maxiter': 150,
                    'maxfun': 400,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20,
                }
            )
            
            if np.all(np.isfinite(result.x)):
                area_candidate = _compute_area_exact(result.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    # Phase 2: Soft-min surrogate with adaptive beta
    # Start with smoother beta, gradually increase for sharper approximation
    if time.time() < deadline - safety_margin - 60.0:
        beta_schedule = [20.0, 50.0, 100.0, 200.0]
        for beta in beta_schedule:
            if time.time() > deadline - safety_margin - 60.0:
                break
            
            try:
                result_soft = minimize(
                    lambda x: _softmin_area_and_grad(x, beta=beta, M=4000)[0],
                    best_x,
                    method='L-BFGS-B',
                    jac=lambda x: _softmin_area_and_grad(x, beta=beta, M=4000)[1],
                    options={
                        'maxiter': 100,
                        'maxfun': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20,
                    }
                )
                
                if np.all(np.isfinite(result_soft.x)):
                    area_candidate = _compute_area_exact(result_soft.x)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = result_soft.x.copy()
                        best_area = area_candidate
            except Exception:
                pass
    
    # Phase 3: Basin hopping with structured perturbations
    if time.time() < deadline - safety_margin - 30.0:
        n_hops = 10
        for hop in range(n_hops):
            if time.time() > deadline - safety_margin - 30.0:
                break
            
            # Perturb current best
            scale = 0.2 / (hop + 1)
            x_perturbed = _structured_perturbation(best_x, rng, scale)
            
            try:
                result_hop = minimize(
                    lambda x: _union_area_and_grad(x, M=4000)[0],
                    x_perturbed,
                    method='L-BFGS-B',
                    jac=lambda x: _union_area_and_grad(x, M=4000)[1],
                    options={
                        'maxiter': 80,
                        'maxfun': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20,
                    }
                )
                
                if np.all(np.isfinite(result_hop.x)):
                    area_candidate = _compute_area_exact(result_hop.x)
                    if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                        best_x = result_hop.x.copy()
                        best_area = area_candidate
            except Exception:
                pass
    
    # Phase 4: Final Nelder-Mead polish at full resolution
    if time.time() < deadline - safety_margin:
        try:
            result2 = minimize(
                _compute_area_exact,
                best_x,
                method='Nelder-Mead',
                options={
                    'maxiter': 300,
                    'maxfev': 300,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True,
                }
            )
            
            if np.all(np.isfinite(result2.x)):
                area_candidate = _compute_area_exact(result2.x)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result2.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    return best_x


# EVOLVE-BLOCK-END
