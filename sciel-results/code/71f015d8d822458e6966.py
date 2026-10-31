import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s buffer

    def project_to_polytope(h, S):
        """Project h onto { x : sum(x) = S, 0 <= x_i <= 1 }"""
        if S < 0 or S > N:
            return np.clip(h, 0, 1)
            
        # Binary search for lambda
        # f(lambda) = sum(clip(h_i - lambda, 0, 1))
        # We want f(lambda) = S
        # f is decreasing.
        lo = np.min(h) - 1.0
        hi = np.max(h) + 1.0
        
        for _ in range(60):
            mid = (lo + hi) / 2.0
            val = np.sum(np.clip(h - mid, 0.0, 1.0))
            if val > S:
                lo = mid
            else:
                hi = mid
                
        lambda_star = (lo + hi) / 2.0
        return np.clip(h - lambda_star, 0.0, 1.0)

    def compute_overlaps(h):
        # O_k = sum_i h_i (1 - h_{i+k}) = N/2 - sum_i h_i h_{i+k}
        overlaps = np.zeros(N)
        for k in range(N):
            shifted = np.roll(h, -k)
            overlaps[k] = N/2 - np.dot(h, shifted)
        return overlaps

    def smooth_max_grad(h, beta):
        """
        Returns (smooth_max_value, gradient)
        """
        overlaps = compute_overlaps(h)
        
        # Stable log-sum-exp
        max_o = np.max(overlaps)
        exps = np.exp(beta * (overlaps - max_o))
        sum_exps = np.sum(exps)
        
        # Weights
        weights = exps / sum_exps
        
        # Smooth max value
        val = max_o + np.log(sum_exps) / beta
        
        # Gradient of each O_k
        # dO_k/dh_j = (1 - h_{j+k}) - h_{j-k}
        # grad = sum_k weights[k] * ( (1 - h_{j+k}) - h_{j-k} )
        
        term_sum = np.zeros(N)
        for k in range(N):
            if weights[k] > 1e-9:
                term_sum += weights[k] * (np.roll(h, -k) + np.roll(h, k))
                
        grad = 1.0 - term_sum
        return val, grad

    best_h = None
    best_score = float('inf')
    
    # Restart strategy
    num_restarts = 20
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Initialize with sinusoid
        freq = 1 + (r % (N//2))
        # Add some noise
        h = 0.5 + 0.4 * np.sin(2 * np.pi * freq * np.arange(N) / N)
        h += 0.05 * np.random.randn(N)
        
        # Project to feasible set
        h = project_to_polytope(h, N/2)
        
        # Optimization loop
        beta = 1.0
        alpha = 0.01
        max_steps = 200
        
        for step in range(max_steps):
            if time.time() > deadline - 5:
                break
                
            val, grad = smooth_max_grad(h, beta)
            
            # Check actual max overlap for early stopping or tracking
            actual_overlaps = compute_overlaps(h)
            current_max = np.max(actual_overlaps)
            
            if current_max < best_score:
                best_score = current_max
                best_h = h.copy()
                
            # Update
            h = h - alpha * grad
            
            # Project back
            h = project_to_polytope(h, N/2)
            
            # Adaptive Beta and Alpha
            # Increase beta to sharpen the max
            beta *= 1.05
            # Decrease alpha slowly
            alpha *= 0.99
            
            if np.max(np.abs(grad)) < 1e-6:
                break

    if best_h is None:
        return np.full(N, 0.5)
        
    # Final polish: ensure exact sum
    best_h = project_to_polytope(best_h, N/2)
    
    return best_h
