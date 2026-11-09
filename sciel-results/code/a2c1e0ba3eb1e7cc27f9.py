# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, basinhopping

def _union_area(x, n=64, M=20000):
    """
    Compute the area of the union of n triangles using the specified quadrature.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    # L_i(y) = x_i + a_i * y
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L along axis 0
    L_sorted = np.sort(L, axis=0)
    
    # Consecutive differences
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    # Union length = w + sum over j of min(d_j, w)
    min_d_w = np.minimum(d, w[np.newaxis, :])
    
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    
    area = np.mean(union_lengths)
    return area

def _union_area_gradient(x, n=64, M=20000):
    """
    Compute the area and its subgradient with respect to x.
    """
    m = np.arange(M)
    y = (m + 0.5) / M
    i_idx = np.arange(n)
    a = (i_idx + 1) / n
    w = (1.0 - y) / n
    
    L = x[:, np.newaxis] + a[:, np.newaxis] * y[np.newaxis, :]
    
    # Sort L and track indices
    sort_idx = np.argsort(L, axis=0)
    L_sorted = np.take_along_axis(L, sort_idx, axis=0)
    
    d = L_sorted[1:, :] - L_sorted[:-1, :]
    
    min_d_w = np.minimum(d, w[np.newaxis, :])
    union_lengths = w[np.newaxis, :] + np.sum(min_d_w, axis=0)
    area = np.mean(union_lengths)
    
    grad = np.zeros(n, dtype=np.float64)
    
    # mask for where d_j < w (active constraints)
    mask = d < w[np.newaxis, :]
    
    # Accumulate gradients
    # For each j in 0..n-2:
    #   if mask[j, m] is True:
    #     grad[sort_idx[j+1, m]] += 1/M
    #     grad[sort_idx[j, m]] -= 1/M
    
    # Vectorized accumulation using np.add.at is slow for large M.
    # We can try to do it in chunks or use a different approach.
    # Given n=64 and M=20000, the loop over j (63 iterations) is acceptable.
    
    for j in range(n - 1):
        mask_j = mask[j, :]
        if np.any(mask_j):
            upper_idx = sort_idx[j + 1, mask_j]
            lower_idx = sort_idx[j, mask_j]
            np.add.at(grad, upper_idx, 1.0 / M)
            np.add.at(grad, lower_idx, -1.0 / M)
            
    return area, grad

def _keich_init(n):
    """
    Generate initial positions based on the Keich construction.
    """
    k = int(np.log2(n))
    j_arr = np.arange(1, k + 1)
    x0 = np.zeros(n, dtype=np.float64)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        x0[i] = np.sum((1 - j_arr) * eps * (2.0 ** -j_arr) / k)
    return x0

def kakeya_needle_2d_positions64() -> np.ndarray:
    """
    Two-dimensional Kakeya-needle area-minimization problem.
    Returns:
        positions: np.ndarray of shape (64,) with the base x-coordinates x_i.
    """
    n = 64
    M = 20000
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin
    
    # 1. Initial Guess
    x0 = _keich_init(n)
    best_x = x0.copy()
    best_area = _union_area(x0, n, M)
    
    # 2. Local Optimization with L-BFGS-B
    def objective(x):
        return _union_area(x, n, M)
        
    def objective_grad(x):
        return _union_area_gradient(x, n, M)
    
    try:
        # Polish the initial guess
        res = minimize(
            objective_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 50, 'gtol': 1e-6}
        )
        if res.fun < best_area:
            best_x = res.x.copy()
            best_area = res.fun
    except Exception:
        pass
    
    # 3. Basin Hopping to escape local minima
    if time.time() < deadline - 120:
        def step(x):
            # Random kick
            return x + 0.5 * np.random.randn(n)
            
        # Basin hopping uses a local minimizer internally
        # We use L-BFGS-B as the local minimizer
        # min_iters and max_iters control the number of basin hops
        # We need to be careful with the time limit inside basinhopping
        # It's safer to implement a manual loop of kick + local minimize
        
        current_x = best_x.copy()
        current_area = best_area
        
        # Simple manual basin hopping loop
        step_size = 1.0
        for i in range(50): # Max 50 hops
            if time.time() > deadline - 30:
                break
                
            # Generate a new candidate
            x_new = current_x + step_size * np.random.randn(n)
            
            # Local minimize from x_new
            try:
                res = minimize(
                    objective_grad,
                    x_new,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 20, 'gtol': 1e-6}
                )
                area_new = res.fun
                x_new = res.x
            except Exception:
                area_new = _union_area(x_new, n, M)
                
            # Acceptance criterion (Metropolis or simple greedy)
            # For this problem, simple greedy (accept if better) is often sufficient
            # but Metropolis can help escape. Let's use a simple greedy with occasional restarts.
            if area_new < best_area:
                best_x = x_new.copy()
                best_area = area_new
                current_x = x_new.copy()
                current_area = area_new
                # Reduce step size on success
                step_size *= 0.9
            else:
                # Keep current, maybe increase step size slightly
                step_size = min(step_size * 1.05, 5.0)
                
    # 4. Final Polish
    if time.time() < deadline - 10:
        try:
            res = minimize(
                objective_grad,
                best_x,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            if res.fun < best_area:
                best_x = res.x.copy()
                best_area = res.fun
        except Exception:
            pass
            
    # Validation
    if not np.all(np.isfinite(best_x)):
        return _keich_init(n)
    if best_area <= 0:
        return _keich_init(n)
        
    return best_x.astype(np.float64)

# EVOLVE-BLOCK-END
