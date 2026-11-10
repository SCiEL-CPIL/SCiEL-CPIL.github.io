# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

def kakeya_needle_2d_positions64() -> np.ndarray:
    n = 64
    M = 20000
    y = (np.arange(M) + 0.5) / M
    a = (np.arange(n) + 1) / n
    w = (1 - y) / n
    
    # Precompute Keich construction for initialization
    k = int(np.log2(n))
    j = np.arange(1, k + 1)
    keich_pos = np.zeros(n)
    for i in range(n):
        eps = np.array(list(str(bin(i))[2:]), dtype=int)
        eps = np.pad(eps, (k - len(eps), 0), "constant")
        keich_pos[i] = np.sum((1 - j) * eps * (2.0 ** -j) / k)
        
    # Shift so x_0 = 0
    keich_pos = keich_pos - keich_pos[0]
    
    def objective_and_grad(x_opt):
        # x_opt is shape (n-1,)
        x_full = np.empty(n)
        x_full[0] = 0.0
        x_full[1:] = x_opt
        
        # L shape (M, n)
        # L[m, i] = x_full[i] + a[i] * y[m]
        L = x_full[None, :] + a[None, :] * y[:, None]
        
        # Sort L along axis 1
        # We need the indices to map gradients back
        idx = np.argsort(L, axis=1)
        L_sorted = np.take_along_axis(L, idx, axis=1)
        
        # Compute gaps D shape (M, n-1)
        D = L_sorted[:, 1:] - L_sorted[:, :-1]
        
        # Compute area
        # union_len_m = w[m] + sum(D_m < w[m]) * D_m + sum(D_m >= w[m]) * w[m]
        # min(D, w) = np.minimum(D, w[:, None])
        min_gaps = np.minimum(D, w[:, None])
        area = np.mean(w + np.sum(min_gaps, axis=1))
        
        # Compute gradient
        # grad_full shape (n,)
        grad_full = np.zeros(n)
        
        # Mask where gap < w
        mask = D < w[:, None] # shape (M, n-1)
        
        # For each gap k (0 to n-2), if mask[m, k] is True:
        # grad[idx[m, k]] -= 1
        # grad[idx[m, k+1]] += 1
        
        # Vectorized scatter
        # We can use np.add.at
        # But idx is (M, n). We need to access idx[:, k] and idx[:, k+1]
        # This loop over n-1 is small (63), so it's fine.
        for k in range(n - 1):
            # mask[:, k] is shape (M,)
            # We only want to update where mask is True
            # left_indices = idx[:, k]
            # right_indices = idx[:, k+1]
            
            # np.add.at(grad_full, indices, values)
            # We need to apply -1 to left_indices where mask is True
            # and +1 to right_indices where mask is True
            
            # Create arrays of indices to update
            # This might be memory heavy if done naively, but M=20000 is small.
            # Better: use boolean indexing? No, grad_full is 1D.
            
            # Efficient way:
            # left_idx = idx[:, k]
            # right_idx = idx[:, k+1]
            # update_mask = mask[:, k]
            
            # np.add.at is the standard way for repeated indices, but here indices are unique per row?
            # No, across rows, indices repeat.
            
            # Let's just do it row by row? No, M=20000.
            # Looping M is 20000 iterations. Inside loop, update 2 elements.
            # 40000 updates. This is fast in Python if vectorized? No, Python loop is slow.
            
            # Let's try np.add.at with full arrays.
            # We need to filter indices by mask.
            # left_to_update = idx[:, k][mask[:, k]]
            # right_to_update = idx[:, k+1][mask[:, k]]
            
            # This creates temporary arrays. Total size ~ M/2.
            # 63 iterations * (M/2) ops. ~ 63 * 10000 = 630,000 ops. Very fast.
            
            left_to_update = idx[:, k][mask[:, k]]
            right_to_update = idx[:, k+1][mask[:, k]]
            
            np.add.at(grad_full, left_to_update, -1.0)
            np.add.at(grad_full, right_to_update, 1.0)
            
        # Scale gradient by 1/M
        grad_full /= M
        
        # Return gradient for x_opt (exclude x_0)
        return area, grad_full[1:]

    # Optimize
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    x0 = keich_pos[1:]
    
    # Use L-BFGS-B
    # Since the problem is convex, we can run until convergence.
    # maxiter should be high enough.
    result = scipy.optimize.minimize(
        objective_and_grad, 
        x0, 
        method='L-BFGS-B', 
        jac=True,
        options={'maxiter': 5000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    x_best = result.x
    x_full_best = np.empty(n)
    x_full_best[0] = 0.0
    x_full_best[1:] = x_best
    
    # Check time
    if time.time() > deadline:
        # If we ran out of time, return the last known best?
        # scipy.optimize.minimize returns the best point found so far in result.x
        # even if it didn't converge perfectly, usually.
        pass
        
    return x_full_best

# EVOLVE-BLOCK-END
