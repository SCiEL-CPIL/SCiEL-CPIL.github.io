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
    # For each active pair (m, j): +1 to order[m, j+1], -1 to order[m, j]
    # Flatten: for all m, j where mask[m,j]:
    #   right_indices = order[m, j+1] -> +1
    #   left_indices = order[m, j] -> -1
    
    # Get indices where mask is True
    m_idx, j_idx = np.where(mask)
    
    if len(m_idx) > 0:
        left_indices = order[m_idx, j_idx]
        right_indices = order[m_idx, j_idx + 1]
        
        # Use bincount for efficient accumulation
        grad_right = np.bincount(right_indices, minlength=n).astype(np.float64)
        grad_left = np.bincount(left_indices, minlength=n).astype(np.float64)
        grad = grad_right - grad_left
    else:
        grad = np.zeros(n, dtype=np.float64)
    
    grad /= M  # mean over quadrature points
    
    return area, grad


def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem for n=64.
    
    Uses the Keich construction as initialization, then refines with L-BFGS-B
    using the analytical gradient of the union area.
    Multiple restarts with random perturbations to escape local minima.
    
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    k = int(np.log2(n))
    
    # Keich construction as initial guess
    j_arr = np.arange(1, k + 1)
    positions_keich = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        positions_keich[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    
    start_time = time.time()
    deadline = start_time + 630.0  # 630 second limit
    safety_margin = 30.0  # seconds before deadline
    
    best_x = positions_keich.copy()
    best_area = _union_area_and_grad(best_x, M=20000)[0]
    
    # Generate multiple initial guesses: Keich + random perturbations
    rng = np.random.default_rng(42)
    initial_guesses = [positions_keich.copy()]
    
    # Add perturbed versions
    for attempt in range(10):
        perturbation_scale = 0.5 / (attempt + 1)
        perturbed = positions_keich + rng.normal(0, perturbation_scale, size=n)
        initial_guesses.append(perturbed)
    
    # Also add some fully random starts
    for attempt in range(5):
        random_start = rng.uniform(-1, 1, size=n)
        initial_guesses.append(random_start)
    
    # Try each initial guess with L-BFGS-B
    for idx, x0 in enumerate(initial_guesses):
        if time.time() > deadline - safety_margin:
            break
        
        try:
            result = minimize(
                lambda x: _union_area_and_grad(x, M=4000),
                x0,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 100,
                    'maxfun': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 20,
                }
            )
            
            if np.all(np.isfinite(result.x)):
                area_candidate, _ = _union_area_and_grad(result.x, M=20000)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    # Final polish with full quadrature on the best solution
    if time.time() < deadline - safety_margin:
        try:
            result2 = minimize(
                lambda x: _union_area_and_grad(x, M=20000),
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 50,
                    'maxfun': 100,
                    'ftol': 1e-14,
                    'gtol': 1e-10,
                    'maxls': 20,
                }
            )
            
            if np.all(np.isfinite(result2.x)):
                area_candidate, _ = _union_area_and_grad(result2.x, M=20000)
                if np.isfinite(area_candidate) and area_candidate > 0 and area_candidate < best_area:
                    best_x = result2.x.copy()
                    best_area = area_candidate
        except Exception:
            pass
    
    return best_x


# EVOLVE-BLOCK-END
