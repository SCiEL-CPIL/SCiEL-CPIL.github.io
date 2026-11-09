import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    # Quadrature heights
    y = (np.arange(M) + 0.5) / M
    w = (1 - y) / n
    
    # a_i = (i+1)/n
    a = (np.arange(n) + 1) / n
    
    # Precompute a_i * y_m for all i, m
    # Shape: (n, M)
    Ay = np.outer(a, y)
    
    def compute_area_and_grad(x):
        # x: (n,)
        # L_i = x_i + a_i * y
        # L shape: (n, M)
        L = x[:, np.newaxis] + Ay
        
        # Sort L at each height
        # idx[m, k] is the index of the k-th smallest L at height m
        idx = np.argsort(L, axis=0)
        # L_sorted[m, k] = L[idx[m, k], m]
        L_sorted = np.take_along_axis(L, idx, axis=0)
        
        # Compute differences between consecutive sorted L's
        # d[m, k] = L_sorted[m, k+1] - L_sorted[m, k] for k=0..n-2
        d = L_sorted[:, 1:] - L_sorted[:, :-1]
        
        # Union length at each height
        # U_m = w_m + sum_k min(w_m, d_m, k)
        # min(w, d) is element-wise
        union_lengths = w + np.sum(np.minimum(w[:, np.newaxis], d), axis=1)
        area = np.mean(union_lengths)
        
        # Gradient
        # grad_i = (1/M) * sum_m ( -1_{L_i - L_prev < w} + 1_{L_next - L_i < w} )
        # We need to know for each i, at each height m, who is its predecessor and successor in the sorted order.
        
        # Initialize gradient
        grad = np.zeros(n)
        
        # For each height m:
        # The term min(w, L_{k+1} - L_k) contributes to the gradient of L_k and L_{k+1}.
        # If L_{k+1} - L_k < w:
        #   d/dL_k (L_{k+1} - L_k) = -1
        #   d/dL_{k+1} (L_{k+1} - L_k) = 1
        # Otherwise, contribution is 0.
        
        # mask[m, k] is True if d[m, k] < w[m]
        mask = d < w[:, np.newaxis]
        
        # For each m and each k where mask[m, k] is True:
        # idx[m, k] is the index of L_k (the smaller one)
        # idx[m, k+1] is the index of L_{k+1} (the larger one)
        # grad[idx[m, k]] -= 1
        # grad[idx[m, k+1]] += 1
        
        # We can use np.add.at for accumulation
        # But we need to do it for all m and k.
        # Let's flatten the masks and indices.
        
        # k indices: 0 to n-2
        k_range = np.arange(n - 1)
        
        # For all m, k where mask is True
        # We want to subtract 1 from grad at index idx[m, k]
        # and add 1 to grad at index idx[m, k+1]
        
        # Create arrays of indices to update
        # m_indices: shape (num_true,)
        # k_indices: shape (num_true,)
        
        # Instead of flattening, we can use broadcasting and add.at?
        # np.add.at(grad, idx[:, :-1][mask], -1)
        # np.add.at(grad, idx[:, 1:][mask], 1)
        
        # Let's check shapes.
        # idx is (M, n)
        # idx[:, :-1] is (M, n-1)
        # idx[:, 1:] is (M, n-1)
        # mask is (M, n-1)
        
        # Extract indices where mask is True
        # This might be slow if done in Python loop, but np.add.at is C-level.
        # However, extracting boolean indices creates a copy.
        # Alternative:
        # contrib_prev = np.where(mask, -1, 0) # (M, n-1)
        # contrib_next = np.where(mask, 1, 0)
        
        # We want to scatter these to grad.
        # grad[idx[m, k]] += contrib_prev[m, k]
        
        # We can use np.bincount if we flatten.
        # flat_idx_prev = idx[:, :-1].ravel()
        # flat_contrib_prev = contrib_prev.ravel()
        # np.add.at(grad, flat_idx_prev, flat_contrib_prev)
        
        flat_idx_prev = idx[:, :-1].ravel()
        flat_idx_next = idx[:, 1:].ravel()
        flat_mask = mask.ravel()
        
        # Only add where mask is True to save time? 
        # np.add.at with 0s is fine but processes all elements.
        # Let's just add the full arrays, it's M*(n-1) ~ 1.2M elements. Fast enough.
        
        np.add.at(grad, flat_idx_prev, np.where(flat_mask, -1.0, 0.0))
        np.add.at(grad, flat_idx_next, np.where(flat_mask, 1.0, 0.0))
        
        grad /= M
        
        return area, grad

    # Initial guess: Keich construction
    def keich_positions(n):
        k = int(np.log2(n))
        j = np.arange(1, k + 1)
        positions = []
        for i in range(n):
            eps = np.array(list(str(bin(i))[2:]), dtype=int)
            eps = np.pad(eps, (k - len(eps), 0), "constant")
            positions.append(np.sum((1 - j) * eps * (2.0 ** -j) / k))
        return np.array(positions, dtype=np.float64)

    x0 = keich_positions(n)
    
    # Other initial guesses
    # 1. Keich
    # 2. Keich + small noise
    # 3. Random uniform in [0, 1)
    # 4. Sorted uniform?
    
    rng = np.random.default_rng(42)
    starts = [x0]
    for _ in range(10):
        starts.append(x0 + rng.uniform(-0.05, 0.05, n))
    for _ in range(5):
        starts.append(rng.uniform(0, 1, n))
        
    best_x = x0
    best_area = np.inf
    
    start_time = time.time()
    time_limit = 550.0 # Leave 50s margin
    
    for i, x_start in enumerate(starts):
        if time.time() - start_time > time_limit:
            break
            
        try:
            # L-BFGS-B
            res = minimize(
                lambda x: compute_area_and_grad(x)[0],
                x_start,
                jac=lambda x: compute_area_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < best_area:
                # Verify and potentially polish
                area, grad = compute_area_and_grad(res.x)
                if area < best_area:
                    best_area = area
                    best_x = res.x.copy()
                    
        except Exception as e:
            pass

    # Final polish on the best solution
    if time.time() - start_time < time_limit - 10:
        try:
            res_final = minimize(
                lambda x: compute_area_and_grad(x)[0],
                best_x,
                jac=lambda x: compute_area_and_grad(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-10}
            )
            if res_final.fun < best_area:
                best_area = res_final.fun
                best_x = res_final.x.copy()
        except Exception:
            pass

    return best_x
