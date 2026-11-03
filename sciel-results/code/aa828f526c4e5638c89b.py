import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Initialize h
    h = np.random.rand(N)
    # Normalize to sum N/2
    h = h / h.sum() * (N / 2.0)
    # Clip to [0,1] and re-normalize? 
    # Better to start with a valid point.
    # Simple valid point: 0.5 everywhere.
    h = np.full(N, 0.5)
    
    # Add some noise to escape symmetry
    h += 0.1 * np.random.randn(N)
    h = np.clip(h, 0, 1)
    # Re-normalize to sum N/2
    # We need to project onto sum=N/2, 0<=h<=1
    # Function to project
    def project(h_raw):
        # Find lambda such that sum(clip(h_raw - lambda, 0, 1)) == N/2
        lo, hi = -1, 2
        for _ in range(50):
            mid = (lo + hi) / 2
            h_proj = np.clip(h_raw - mid, 0, 1)
            s = h_proj.sum()
            if s > N / 2.0:
                lo = mid
            else:
                hi = mid
        return np.clip(h_raw - (lo+hi)/2, 0, 1)
        
    h = project(h)
    
    # NAG parameters
    lr = 0.01
    beta = 0.9
    tau = 10.0 # Temperature for smooth max
    max_iter = 2000
    
    v = np.zeros(N)
    prev_grad = np.zeros(N)
    
    def compute_E_and_grad(h):
        # E_k = sum_i (h_i + h_{i-k} - 1)^2
        # Grad = sum_k w_k * grad_E_k
        # grad_E_k[i] = 2(h_i + h_{i-k} - 1) + 2(h_i + h_{i+k} - 1)
        
        E_vals = np.zeros(N-1)
        grad_E = np.zeros((N-1, N))
        
        for k in range(1, N):
            # Circular shift
            h_shifted = np.roll(h, -k) # h_{i-k} is at index i in shifted? 
            # If we want h_{i-k}, and we have h array, 
            # h_shifted[i] = h[(i-k)%N]
            # np.roll(h, -k) shifts right by k? 
            # np.roll([1,2,3,4], -1) -> [2,3,4,1]. 
            # So h_shifted[i] = h[i+1]. This is h_{i+1}.
            # We want h_{i-k}.
            # np.roll(h, k) shifts left? No, np.roll(h, k) shifts right by k?
            # Let's check: np.roll([1,2,3,4], 1) -> [4,1,2,3]. 
            # So h_shifted[i] = h[i-1]. This is h_{i-1}.
            # So for k=1, we want h_{i-1}. np.roll(h, 1) gives h_{i-1}.
            # So h_minus_k = np.roll(h, k)
            
            h_minus_k = np.roll(h, k)
            h_plus_k = np.roll(h, -k)
            
            diff = h + h_minus_k - 1.0
            E_vals[k-1] = np.sum(diff**2)
            
            # Gradient of E_k w.r.t h
            # d/dh_i (h_i + h_{i-k} - 1)^2 = 2(h_i + h_{i-k} - 1)
            # h_i also appears in term for index i+k: (h_{i+k} + h_i - 1)^2
            # So grad[i] = 2(h_i + h_{i-k} - 1) + 2(h_i + h_{i+k} - 1)
            
            grad_E[k-1, :] = 2.0 * diff + 2.0 * (h + h_plus_k - 1.0)
            
        return E_vals, grad_E

    best_h = h.copy()
    best_score = np.inf
    
    for i in range(max_iter):
        if time.time() - start_time > timeout:
            break
            
        E_vals, grad_E = compute_E_and_grad(h)
        
        # Smooth max weights
        # Avoid overflow
        max_E = np.max(E_vals)
        exp_E = np.exp(tau * (E_vals - max_E))
        sum_exp = np.sum(exp_E)
        if sum_exp == 0:
            w = np.ones(N-1) / (N-1)
        else:
            w = exp_E / sum_exp
            
        # Weighted gradient
        grad_obj = np.sum(w[:, np.newaxis] * grad_E, axis=0)
        
        # Penalty gradient
        # Sum constraint: sum(h) - N/2
        penalty_sum = 10.0
        grad_pen = penalty_sum * np.ones(N)
        
        # Bounds penalty:
        # h < 0: grad = 2*lambda*h (if h<0)
        # h > 1: grad = 2*lambda*(h-1) (if h>1)
        penalty_bound = 10.0
        grad_pen += penalty_bound * np.where(h < 0, 2*h, 0)
        grad_pen += penalty_bound * np.where(h > 1, 2*(h-1), 0)
        
        grad_total = grad_obj + grad_pen
        
        # NAG update
        # v = beta * v + grad_total
        # h = h - lr * v
        # Standard NAG:
        # h_new = h - lr * (grad_total + beta * (grad_total - prev_grad))
        # Or:
        # v = beta * v + grad_total
        # h = h - lr * v
        # This is momentum. NAG is slightly different but similar for fixed step.
        # Let's use standard momentum for simplicity and stability.
        
        v = beta * v + grad_total
        h_new = h - lr * v
        
        # Project h_new
        h_new = project(h_new)
        
        h = h_new
        prev_grad = grad_total
        
        # Evaluate raw score (max E_k)
        E_vals, _ = compute_E_and_grad(h)
        current_score = np.max(E_vals)
        
        if current_score < best_score:
            best_score = current_score
            best_h = h.copy()
            
    return best_h
