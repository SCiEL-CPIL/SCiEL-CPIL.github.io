import numpy as np
import time

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Initialize with a spread-out pattern to avoid immediate clustering
    # Using a "Barker-like" or pseudo-random binary sequence initialized to 1s
    # but with some zeros to break symmetry
    h = np.ones(N)
    # Randomly zero out ~30% of entries to encourage spread
    mask = np.random.random(N) < 0.3
    h[mask] = 0.0
    
    # Ensure at least one non-zero
    if np.sum(h) == 0:
        h[N//2] = 1.0
        
    # Normalize to max norm 1
    h = h / np.max(h)
    
    def compute_ratio(h):
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        l_inf = np.max(h)
        if l1 < 1e-15 or l_inf < 1e-15:
            return 0.0
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15:
            return 0.0
        return (l2_sq) / (l1 * l_inf * max_conv)

    best_h = h.copy()
    best_ratio = compute_ratio(best_h)
    
    # Spectral Water-Filling / Gradient Descent on negative log-ratio
    # We want to maximize R = ||h||_2^2 / (||h||_1 ||h||_inf max(h*h))
    # Taking log: log(R) = 2 log(||h||_2) - log(||h||_1) - log(||h||_inf) - log(max(h*h))
    # Since we normalize h to max=1, ||h||_inf is constant (1).
    # So we maximize 2 log(||h||_2) - log(||h||_1) - log(max(h*h))
    # Note: ||h||_2^2 = sum h_i^2. ||h||_1 = sum h_i.
    # Let f(h) = - (2 log(sum h_i^2) - log(sum h_i) - log(max_conv))
    # We minimize f(h).
    
    # Gradient approximation via finite differences is too slow.
    # Use analytical gradient where possible.
    # max_conv = max_k sum_i h_i h_{i+k} (with appropriate shifts for full conv)
    # Let M = max_conv. Let S be the set of indices k where h*h reaches M.
    # d(M)/dh_i = sum_{k in S} (h_{i-k} + h_{i+k}) (adjusting for boundaries)
    # Actually, for full convolution c = h*h, c_m = sum_i h_i h_{m-i}.
    # If m is the argmax, d(c_m)/dh_i = h_{m-i} + h_{m-i} (if i != m-i) or 2h_i.
    # Wait, c_m = sum_i h_i h_{m-i}.
    # d(c_m)/dh_i = h_{m-i} + h_{m-i} (chain rule: term i and term m-i)
    # If i == m-i (i.e., 2i=m), derivative is 2h_i. Otherwise h_{m-i} + h_{m-i} = 2 h_{m-i}? No.
    # Let's be precise. c_m = sum_j h_j h_{m-j}.
    # d(c_m)/dh_i = h_{m-i} (from j=i) + h_{m-i} (from j=m-i, if i != m-i).
    # If i == m-i, term is h_i^2, derivative is 2h_i.
    # So d(c_m)/dh_i = h_{m-i} + h_{m-i} if i != m-i, else 2h_i.
    # Actually, if we sum over all j, the term involving h_i is h_i h_{m-i} (when j=i) and h_{m-i} h_i (when j=m-i).
    # If i != m-i, these are two distinct terms in the sum, so derivative is h_{m-i} + h_{m-i} = 2 h_{m-i}? No.
    # The term is h_j h_{m-j}.
    # If j=i, term is h_i h_{m-i}. Deriv wrt h_i is h_{m-i}.
    # If j=m-i, term is h_{m-i} h_i. Deriv wrt h_i is h_{m-i}.
    # So total derivative is 2 h_{m-i} if i != m-i.
    # If i = m-i, term is h_i^2. Deriv is 2 h_i.
    # So d(c_m)/dh_i = h_{m-i} + h_{m-i} (with h_{m-i}=h_i in the second case? No, h_{m-i} is just the value).
    # Let's just use numerical gradient for the max_conv term if analytical is tricky, 
    # but N=50 is small enough for simple updates.
    
    # Strategy: Iterative Water Filling
    # At each step, compute the autoconvolution.
    # Identify the peak location(s).
    # Reduce the height of elements contributing to the peak.
    # Increase the height of elements contributing to "valleys" or non-peak parts.
    
    # Simpler robust approach: Coordinate Descent on the objective function.
    # Objective: Maximize J(h) = log(sum h^2) - log(sum h) - log(max_conv)
    # (ignoring constants and inf-norm normalization which is 1)
    
    def objective(h):
        l2_sq = np.sum(h**2)
        l1 = np.sum(h)
        if l1 < 1e-15: return -np.inf
        h_conv = np.convolve(h, h, mode='full')
        max_conv = np.max(h_conv)
        if max_conv < 1e-15: return -np.inf
        return np.log(l2_sq) - np.log(l1) - np.log(max_conv)
    
    current_obj = objective(h)
    
    # Adaptive step size
    step_size = 0.05
    no_improve_count = 0
    
    while time.time() < deadline:
        # Compute gradients numerically for each coordinate
        # N=50, so 50 evaluations per step.
        # Each eval is O(N^2) for conv. 50*2500 = 125k ops. Very fast.
        
        grad = np.zeros(N)
        for i in range(N):
            h_plus = h.copy()
            h_minus = h.copy()
            
            # Move i up
            new_val_plus = np.clip(h[i] + step_size, 0, 1)
            h_plus[i] = new_val_plus
            # Renormalize to max 1? 
            # The problem statement says "nonnegative step heights". 
            # The score formula is ||h||_2^2 / (||h||_1 ||h||_inf max_conv).
            # This ratio is scale-invariant! 
            # (ah)^2 / (a||h||_1 * a||h||_inf * a^2 max(h*h)) = h^2 / (||h||_1 ||h||_inf max(h*h))
            # So we don't strictly need to renormalize to max 1 at every step, 
            # but it helps with numerical stability and keeps values in [0,1].
            # Let's keep them in [0,1] without forcing max to 1, 
            # because the ratio is invariant. 
            # However, if all become 0, we fail.
            
            obj_plus = objective(h_plus)
            
            # Move i down
            new_val_minus = np.clip(h[i] - step_size, 0, 1)
            h_minus[i] = new_val_minus
            obj_minus = objective(h_minus)
            
            # Central difference or forward? 
            # Forward is faster. Let's use forward difference from current h.
            # Actually, let's just try moving each coordinate in the direction that improves objective.
            # This is a greedy coordinate ascent.
            
        # Instead of full gradient, let's do greedy coordinate ascent with random order
        # This avoids calculating all gradients if one move is good.
        
        # Shuffle indices
        indices = np.random.permutation(N)
        improved = False
        
        for i in indices:
            if time.time() > deadline:
                break
                
            old_val = h[i]
            
            # Try a few random perturbations
            best_local_val = old_val
            best_local_obj = current_obj
            
            for _ in range(5):
                delta = np.random.uniform(-step_size, step_size)
                new_val = np.clip(old_val + delta, 0, 1)
                if new_val == old_val:
                    continue
                    
                h[i] = new_val
                obj_new = objective(h)
                
                if obj_new > best_local_obj:
                    best_local_obj = obj_new
                    best_local_val = new_val
                h[i] = old_val # revert to try next
            
            if best_local_val != old_val:
                h[i] = best_local_val
                current_obj = best_local_obj
                improved = True
                # Update best if better than global best
                ratio = compute_ratio(h)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_h = h.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        
        if not improved:
            no_improve_count += 1
            # Reduce step size if stuck
            step_size *= 0.9
            if step_size < 1e-4:
                # Reset step size and randomize slightly
                step_size = 0.1
                h = h + np.random.uniform(-0.01, 0.01, N)
                h = np.clip(h, 0, 1)
                # Re-normalize to avoid all zeros
                if np.max(h) == 0:
                    h[N//2] = 1.0
                current_obj = objective(h)
        else:
            # Slowly decay step size to converge
            step_size *= 0.995
            if step_size < 1e-3:
                step_size = 1e-3

    return list(best_h)

