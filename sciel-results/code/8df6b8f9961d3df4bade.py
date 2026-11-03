"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    """
    Construct the step heights h to minimize max overlap of h with shifted 1-h.
    
    Theoretical analysis:
    C(k) = sum_i h_i * (1 - h_{i+k})
    Sum over all shifts k=1..N-1: Sum_k C(k) = N * Sum_i h_i - (Sum_i h_i)^2 = N*(N/2) - (N/2)^2 = N^2/4
    Therefore average C(k) = N^2/4 / (N-1), so max_k C(k) >= N^2/(4(N-1)).
    
    Uniform h_i = 0.5 achieves C(k) = N * 0.25 = N/4 for every k.
    Note: N/4 vs N^2/(4(N-1)): N/4 = N(N-1)/(4(N-1)) + N/(4(N-1)), so uniform is slightly above the lower bound.
    
    We use projected gradient descent on a smooth log-sum-exp surrogate
    of the max-over-shifts objective, with careful handling of the sum constraint.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill
    
    h = np.full(N, 0.5)
    
    def compute_max_overlap(h_vec):
        """Compute max over all cyclic shifts k of sum_i h_i * (1 - h_{i+k})."""
        max_c = 0.0
        for k in range(1, N):
            shifted = np.roll(h_vec, k)
            c = np.dot(h_vec, 1.0 - shifted)
            if c > max_c:
                max_c = c
        return max_c
    
    def compute_all_c(h_vec):
        """Compute all C(k) values for k=1..N-1."""
        c_vals = np.zeros(N - 1)
        for k in range(1, N):
            shifted = np.roll(h_vec, k)
            c_vals[k - 1] = np.dot(h_vec, 1.0 - shifted)
        return c_vals
    
    def project_to_simplex(h_vec):
        """Project h_vec onto the set {x in [0,1]^N : sum(x) = N/2}."""
        h = np.clip(h_vec, 0.0, 1.0)
        current_sum = np.sum(h)
        target_sum = N / 2.0
        residual = target_sum - current_sum
        
        if abs(residual) < 1e-15:
            return h
        
        # Iterative projection to handle bounds
        for _ in range(10):
            if residual > 0:
                # Need to add mass; prefer elements with room (h < 1)
                mask = h < 1.0 - 1e-12
            else:
                # Need to remove mass; prefer elements with room (h > 0)
                mask = h > 1e-12
            
            if not np.any(mask):
                break
            
            n_mask = np.sum(mask)
            # Distribute residual, but don't overshoot bounds
            per_element = residual / n_mask
            if residual > 0:
                # Can't add more than (1 - h_i) to each
                max_add = 1.0 - h[mask]
                actual_add = np.minimum(per_element, max_add)
                h[mask] += actual_add
                new_residual = residual - np.sum(actual_add)
            else:
                # Can't remove more than h_i from each
                max_remove = h[mask]
                actual_remove = np.minimum(-per_element, max_remove)
                h[mask] -= actual_remove
                new_residual = residual + np.sum(actual_remove)
            
            if abs(new_residual) < 1e-15:
                break
            residual = new_residual
        
        # Final clip and exact sum fix
        h = np.clip(h, 0.0, 1.0)
        final_residual = target_sum - np.sum(h)
        if abs(final_residual) > 1e-14:
            # Very small adjustment spread across all elements
            h += final_residual / N
            h = np.clip(h, 0.0, 1.0)
        
        return h
    
    # Verify uniform solution
    uniform_score = compute_max_overlap(h)
    best_h = h.copy()
    best_score = uniform_score
    
    # Projected gradient descent on smooth surrogate
    beta = 50.0  # temperature for soft-max (higher = closer to true max)
    lr = 0.005
    max_iterations = 20000
    
    for it in range(max_iterations):
        if time.time() > deadline - 5.0:
            break
        
        # Compute all C(k) values
        c_vals = compute_all_c(h)
        
        # Soft-max surrogate: log(sum(exp(beta * C_k))) / beta
        max_c = np.max(c_vals)
        log_sum_exp = max_c + np.log(np.sum(np.exp(beta * (c_vals - max_c))))
        surrogate = log_sum_exp / beta
        
        # Gradient of surrogate w.r.t. h
        # weights_k = exp(beta * C_k) / sum_j exp(beta * C_j)
        weights = np.exp(beta * (c_vals - max_c))
        weights /= np.sum(weights)
        
        grad = np.zeros(N)
        for k in range(1, N):
            if weights[k - 1] < 1e-15:
                continue
            # dC(k)/dh_i = (1 - h_{(i+k)%N}) - h_{(i-k)%N}
            idx_plus = (np.arange(N) + k) % N
            idx_minus = (np.arange(N) - k) % N
            grad += weights[k - 1] * ((1.0 - h[idx_plus]) - h[idx_minus])
        
        # Gradient step
        h_new = h - lr * grad
        
        # Project to feasible set
        h_new = project_to_simplex(h_new)
        
        # Check if we improved
        score = compute_max_overlap(h_new)
        if score < best_score - 1e-12:
            best_score = score
            best_h = h_new.copy()
            h = h_new
        else:
            # No improvement; try smaller step
            h = h_new  # still move, but don't update best
        
        # If we've hit the theoretical lower bound, stop
        if best_score <= N / 4.0 + 1e-10:
            break
        
        # Adaptive learning rate: reduce if no progress for a while
        if it % 1000 == 0 and it > 0:
            lr *= 0.95
            beta *= 1.2  # sharpen the surrogate over time
    
    # Final polish: try a few more iterations with very small lr
    if time.time() < deadline - 10.0:
        lr_final = 1e-5
        for it in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            c_vals = compute_all_c(best_h)
            max_c = np.max(c_vals)
            weights = np.exp(beta * (c_vals - max_c))
            weights /= np.sum(weights)
            
            grad = np.zeros(N)
            for k in range(1, N):
                if weights[k - 1] < 1e-15:
                    continue
                idx_plus = (np.arange(N) + k) % N
                idx_minus = (np.arange(N) - k) % N
                grad += weights[k - 1] * ((1.0 - best_h[idx_plus]) - best_h[idx_minus])
            
            h_trial = best_h - lr_final * grad
            h_trial = project_to_simplex(h_trial)
            
            score = compute_max_overlap(h_trial)
            if score < best_score - 1e-12:
                best_score = score
                best_h = h_trial
    
    # Ensure best solution is valid
    best_h = np.clip(best_h, 0.0, 1.0)
    current_sum = np.sum(best_h)
    if abs(current_sum - N / 2.0) > 1e-12:
        # Fall back to uniform if sum is off
        best_h = np.full(N, 0.5)
    
    assert np.all(best_h >= -1e-12) and np.all(best_h <= 1.0 + 1e-12)
    assert abs(np.sum(best_h) - N / 2.0) < 1e-10
    best_h = np.clip(best_h, 0.0, 1.0)
    return best_h
# EVOLVE-BLOCK-END
