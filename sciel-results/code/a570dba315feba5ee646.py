import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Initialize x
    # Try a few different initializations
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # We want to minimize max overlap O_k = N/4 - X_k
    # Equivalent to maximizing min X_k
    # We use soft-min on X_k
    
    # Initial guesses
    inits = [
        np.zeros(N),
        np.sin(2 * np.pi * np.arange(N) / N),
        np.random.uniform(-0.5, 0.5, N),
        np.concatenate([np.ones(N//2), -np.ones(N//2)]) # Square wave
    ]
    
    for x0 in inits:
        x = x0.copy()
        # Normalize to sum 0 and scale to max amplitude 0.5
        x = x - np.mean(x)
        if np.max(np.abs(x)) > 0:
            x = x / np.max(np.abs(x)) * 0.5
        
        # Gradient Descent on Soft-Min
        beta = 10.0
        alpha = 0.05
        for i in range(10000):
            if time.time() > deadline:
                break
            
            # Compute X_k
            # X_k = sum x_i x_{i+k}
            # Use FFT for speed? N=100 is small, direct is fine.
            X = np.array([np.sum(x * np.roll(x, -k)) for k in range(N)])
            
            # Soft min loss
            # We want to maximize min X_k, so minimize -min X_k
            # Soft min of X_k is -log-sum-exp(beta * X_k)
            # Gradient of -log(sum exp(beta X_k)) wrt x
            # = - (1/sum) * sum exp(beta X_k) * beta * dX_k/dx
            # dX_k/dx = 2 * (C_k x) where C_k is circulant matrix
            # This is expensive to compute explicitly.
            # dX_k/dx_i = x_{i+k} + x_{i-k}
            
            # Let's compute gradient directly
            grad = np.zeros(N)
            exp_weights = np.exp(beta * X)
            weights = exp_weights / np.sum(exp_weights)
            
            # grad = - sum_k weights_k * beta * dX_k/dx
            # dX_k/dx = x shifted by k + x shifted by -k
            for k in range(N):
                if weights[k] > 1e-9:
                    # dX_k/dx_i = x_{(i+k)%N} + x_{(i-k)%N}
                    dx = np.roll(x, -k) + np.roll(x, k)
                    grad -= weights[k] * beta * dx
            
            # Update
            x = x - alpha * grad
            
            # Project to sum 0
            x = x - np.mean(x)
            # Project to bounds [-0.5, 0.5]
            x = np.clip(x, -0.5, 0.5)
            # Re-project to sum 0 (approximate)
            # Iterative projection
            for _ in range(10):
                x = x - np.mean(x)
                x = np.clip(x, -0.5, 0.5)
            
            # Check convergence
            if i > 100 and np.allclose(x, x0):
                 break
                 
        # Calculate final score
        h = 0.5 + x
        # Ensure sum is exactly N/2
        # Adjust slightly if needed? 
        # The projection ensures sum 0 for x, so sum h = N/2.
        
        X_final = np.array([np.sum(h * (1 - np.roll(h, -k))) for k in range(N)])
        # Overlap is sum h_i (1-h_{i+k})
        # Wait, my X was sum x_i x_{i+k}.
        # O_k = N/4 - sum x_i x_{i+k}
        # Let's compute O_k directly to be safe.
        overlaps = []
        for k in range(N):
            o = np.sum(h * (1 - np.roll(h, -k)))
            overlaps.append(o)
        max_overlap = max(overlaps)
        
        if max_overlap < best_score:
            best_score = max_overlap
            best_h = h.copy()
            
    return best_h
