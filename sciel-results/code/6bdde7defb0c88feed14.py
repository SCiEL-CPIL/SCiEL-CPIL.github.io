import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    N = 100
    
    # Start with uniform solution (known to be optimal for symmetric case)
    h = np.full(N, 0.5)
    
    # Compute overlap function for a given h
    def compute_overlap(h):
        """Compute the maximum overlap of h with shifted 1-h."""
        # h is the step heights, 1-h is the complement
        # We need max over shifts k of sum_i h_i * (1-h)_{i-k}
        # This is equivalent to max over k of (sum h) - autocorrelation-like term
        
        # Using FFT for efficient computation
        n = len(h)
        # Pad to avoid circular convolution issues
        size = 2 * n
        
        H = np.fft.rfft(h, n=size)
        H_conj = np.fft.rfft(1 - h, n=size)
        
        # Cross-correlation: sum_i h_i * (1-h)_{i-k} for each shift k
        cross_corr = np.fft.irfft(H * H_conj.conj(), n=size)[:n]
        
        # The overlap at shift k is cross_corr[k]
        # We want to minimize the maximum of these
        return np.max(cross_corr)
    
    # Smooth max using log-sum-exp
    def smooth_max(x, beta=10.0):
        """Smooth approximation of max using log-sum-exp."""
        x_max = np.max(x)
        return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta
    
    # Objective function: smooth max of overlaps
    def objective(h):
        overlaps = compute_overlap_array(h)
        return smooth_max(overlaps, beta=20.0)
    
    def compute_overlap_array(h):
        """Compute all overlap values for a given h."""
        n = len(h)
        size = 2 * n
        
        H = np.fft.rfft(h, n=size)
        H_conj = np.fft.rfft(1 - h, n=size)
        
        cross_corr = np.fft.irfft(H * H_conj.conj(), n=size)[:n]
        return cross_corr
    
    # Gradient descent with projected constraints
    def project(h):
        """Project h onto the feasible set: 0 <= h_i <= 1, sum(h) = N/2."""
        h = np.clip(h, 0, 1)
        # Adjust to satisfy sum constraint
        current_sum = np.sum(h)
        target_sum = N / 2.0
        if current_sum > 0:
            h = h + (target_sum - current_sum) / N
            h = np.clip(h, 0, 1)
            # Final adjustment
            diff = target_sum - np.sum(h)
            if abs(diff) > 1e-10:
                # Distribute difference evenly
                h = h + diff / N
                h = np.clip(h, 0, 1)
        return h
    
    # Gradient of smooth max
    def gradient(h, beta=20.0):
        overlaps = compute_overlap_array(h)
        x_max = np.max(overlaps)
        weights = np.exp(beta * (overlaps - x_max))
        weights = weights / np.sum(weights)
        
        # Gradient of overlap with respect to h
        # overlap_k = sum_i h_i * (1-h)_{i-k}
        # d(overlap_k)/d(h_j) = (1-h)_{j-k} - h_{j-k}
        
        grad = np.zeros(N)
        for k in range(N):
            w = weights[k]
            # For each j, contribution is w * ((1-h)[(j-k)%N] - h[(j-k)%N])
            shifted_1mh = np.roll(1 - h, k)
            shifted_h = np.roll(h, k)
            grad += w * (shifted_1mh - shifted_h)
        
        return grad
    
    # Try gradient descent
    best_h = h.copy()
    best_obj = objective(h)
    
    lr = 0.01
    n_iter = 5000
    
    for it in range(n_iter):
        if time.time() > deadline:
            break
        
        g = gradient(h, beta=20.0)
        h_new = h - lr * g
        h_new = project(h_new)
        
        obj_new = objective(h_new)
        if obj_new < best_obj:
            best_obj = obj_new
            best_h = h_new.copy()
        
        h = h_new
        
        # Adaptive learning rate
        if it % 500 == 0 and it > 0:
            lr *= 0.9
    
    # Final polish: use scipy if available
    try:
        from scipy.optimize import minimize
        
        def obj_scipy(h_flat):
            h_arr = np.array(h_flat)
            h_arr = np.clip(h_arr, 0, 1)
            # Enforce sum constraint approximately
            h_arr = h_arr + (N/2 - np.sum(h_arr)) / N
            h_arr = np.clip(h_arr, 0, 1)
            return objective(h_arr)
        
        # Use SLSQP with constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2}]
        bounds = [(0, 1)] * N
        
        res = minimize(obj_scipy, best_h, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options={'maxiter': 100, 'ftol': 1e-12})
        
        if res.success and time.time() < deadline:
            h_opt = np.clip(res.x, 0, 1)
            h_opt = h_opt + (N/2 - np.sum(h_opt)) / N
            h_opt = np.clip(h_opt, 0, 1)
            
            if objective(h_opt) < best_obj:
                best_h = h_opt.copy()
    except:
        pass
    
    # Ensure exact sum
    best_h = np.clip(best_h, 0, 1)
    diff = N/2 - np.sum(best_h)
    if abs(diff) > 1e-15:
        # Distribute correction
        indices = np.argsort(best_h)
        for i in indices:
            if diff > 0:
                add = min(diff, 1 - best_h[i])
                best_h[i] += add
                diff -= add
            else:
                sub = min(-diff, best_h[i])
                best_h[i] -= sub
                diff += sub
            if abs(diff) < 1e-15:
                break
    
    return best_h
# EVOLVE-BLOCK-END
