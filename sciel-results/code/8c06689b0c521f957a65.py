import numpy as np
from scipy.optimize import minimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    k = int(np.log2(n))
    
    # Generate Keich positions as initial guess
    j = np.arange(1, k + 1)
    positions = []
    for i in range(n):
        bits = [int(b) for b in bin(i)[2:]]
        bits = [0] * (k - len(bits)) + bits
        eps_seed = np.array(bits, dtype=float)
        pos = np.sum((1 - j) * eps_seed * (2.0 ** -j) / k)
        positions.append(pos)
    x0 = np.array(positions, dtype=np.float64)
    
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n

    def compute_area_exact(x):
        L = x[None, :] + a[None, :] * y[:, None]
        
        order = np.argsort(L, axis=1)
        L_sorted = np.take_along_axis(L, order, axis=1)
        
        diffs = np.diff(L_sorted, axis=1)
        mins = np.minimum(diffs, w[:, None])
        union_lengths = w + np.sum(mins, axis=1)
        area = np.mean(union_lengths)
        
        return area

    def compute_area_smooth_and_grad(x, beta=50.0):
        L = x[None, :] + a[None, :] * y[:, None]
        
        order = np.argsort(L, axis=1)
        L_sorted = np.take_along_axis(L, order, axis=1)
        
        diffs = np.diff(L_sorted, axis=1)
        w_col = w[:, None]
        
        # Smooth min(diffs, w)
        # min(d, w) = 0.5 * (d + w - |d - w|)
        # |d - w| approx sqrt((d-w)^2 + eps^2)
        eps = 1.0 / beta
        diff_w = diffs - w_col
        abs_diff_w = np.sqrt(diff_w**2 + eps**2)
        smooth_mins = 0.5 * (diffs + w_col - abs_diff_w)
        
        union_lengths = w + np.sum(smooth_mins, axis=1)
        area = np.mean(union_lengths)
        
        # Gradient
        # d(area)/d(L_i) involves chain rule through smooth_min
        # d(smooth_min)/d(diffs) = 0.5 * (1 - d(abs_diff_w)/d(diffs))
        # d(abs_diff_w)/d(diffs) = diff_w / abs_diff_w
        # So d(smooth_min)/d(diffs) = 0.5 * (1 - diff_w / abs_diff_w)
        
        grad_mins = 0.5 * (1.0 - diff_w / abs_diff_w)
        
        # grad_mins has shape (M, n-1)
        # We need to map this back to grad_L which has shape (M, n)
        # diffs[j] = L_sorted[j+1] - L_sorted[j]
        # d(smooth_mins[j])/d(L_sorted[j+1]) = grad_mins[j]
        # d(smooth_mins[j])/d(L_sorted[j]) = -grad_mins[j]
        
        grad_L_sorted = np.zeros((M, n))
        # Add contribution from right side of diff
        grad_L_sorted[:, 1:] += grad_mins
        # Add contribution from left side of diff
        grad_L_sorted[:, :-1] -= grad_mins
        
        # Map back to original indices
        grad_L = np.zeros((M, n))
        # grad_L[m, order[m, j]] = grad_L_sorted[m, j]
        # Using np.add.at for safety, though order is a permutation so it's a direct assignment if we invert
        # Inverse permutation
        inv_order = np.empty_like(order)
        np.put_along_axis(inv_order, order, np.arange(n)[None, :], axis=1)
        
        # Actually, it's easier to scatter:
        # grad_L[m, idx] += grad_L_sorted[m, pos]
        # We can use np.take_along_axis in reverse? No.
        # Let's use advanced indexing.
        # grad_L[m, order[m, :]] = grad_L_sorted[m, :]
        grad_L = np.take_along_axis(grad_L_sorted, np.argsort(order, axis=1), axis=1)
        # Wait, if order[m] is the permutation such that L_sorted[m] = L[m, order[m]],
        # then to get grad_L, we need to place grad_L_sorted[m, j] into grad_L[m, order[m, j]].
        # This is a scatter operation.
        # grad_L[m, order[m, j]] = grad_L_sorted[m, j]
        
        # Let's re-verify the mapping.
        # We have gradients with respect to sorted values.
        # We need gradients with respect to original values.
        # Let P be the permutation matrix such that L_sorted = L * P (conceptually).
        # Then dL_sorted = dL * P.
        # grad_L = grad_L_sorted * P^T.
        # In array terms: grad_L[m, order[m, j]] = grad_L_sorted[m, j].
        
        # Efficient scatter:
        # We can create an array of indices.
        # But simple loop or advanced indexing:
        # grad_L = np.zeros((M, n))
        # np.put_along_axis(grad_L, order, grad_L_sorted, axis=1)
        
        grad_L = np.zeros((M, n))
        np.put_along_axis(grad_L, order, grad_L_sorted, axis=1)
        
        grad_x = np.mean(grad_L, axis=0)
        
        return area, grad_x

    start_time = time.time()
    deadline = start_time + 590
    
    best_x = x0.copy()
    best_area = compute_area_exact(x0)
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    num_starts = 10
    
    for s in range(num_starts):
        if time.time() > deadline:
            break
            
        if s == 0:
            current_x = x0.copy()
        else:
            # Perturb the best found so far or the initial guess?
            # Perturbing the best found so far helps explore the local neighborhood more deeply.
            # Perturbing the initial guess helps find different basins.
            # Let's alternate or just perturb the initial guess with increasing noise.
            noise_scale = 0.05 * (s + 1)
            current_x = x0 + rng.normal(0, noise_scale, n)
            
        try:
            # Phase 1: Smooth optimization
            res1 = minimize(
                lambda x, b=50.0: compute_area_smooth_and_grad(x, b)[0],
                current_x,
                jac=lambda x, b=50.0: compute_area_smooth_and_grad(x, b)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 20,
                }
            )
            
            if time.time() > deadline:
                if res1.fun < best_area: # Note: res1.fun is smooth area, need exact comparison
                    # Check exact area
                    exact_area = compute_area_exact(res1.x)
                    if exact_area < best_area:
                        best_x = res1.x.copy()
                        best_area = exact_area
                continue

            # Phase 2: Polish with exact gradient (subgradient)
            # We need the exact gradient function.
            def compute_area_exact_and_grad(x):
                L = x[None, :] + a[None, :] * y[:, None]
                
                order = np.argsort(L, axis=1)
                L_sorted = np.take_along_axis(L, order, axis=1)
                
                diffs = np.diff(L_sorted, axis=1)
                mins = np.minimum(diffs, w[:, None])
                union_lengths = w + np.sum(mins, axis=1)
                area = np.mean(union_lengths)
                
                active = diffs < w[:, None]
                
                grad = np.zeros((M, n))
                m_idx, j_idx = np.where(active)
                if len(m_idx) > 0:
                    left_tri = order[m_idx, j_idx]
                    right_tri = order[m_idx, j_idx + 1]
                    np.add.at(grad, (m_idx, left_tri), -1.0)
                    np.add.at(grad, (m_idx, right_tri), 1.0)
                
                grad_x = np.mean(grad, axis=0)
                
                return area, grad_x

            res2 = minimize(
                lambda x: compute_area_exact_and_grad(x)[0],
                res1.x,
                jac=lambda x: compute_area_exact_and_grad(x)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 100,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 10,
                }
            )
            
            final_area = compute_area_exact(res2.x)
            if final_area < best_area:
                best_x = res2.x.copy()
                best_area = final_area
                
        except Exception:
            pass

    return best_x

# EVOLVE-BLOCK-END
