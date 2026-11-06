import numpy as np
import time

N = 400

def construct():
    # Start with uniform
    h = np.ones(N)
    h /= np.sum(h) # Sum = 1
    
    # Use a smooth max approximation
    # Objective: min 2 * N * softmax(h*h)
    # Constraint: sum(h) = 1
    
    # Gradient Descent
    # V = 2 * N * M / S^2
    # If S=1, V = 2 * N * M
    # dV/dh = 2 * N * dM/dh
    
    # M = max(h*h)
    # Softmax M_alpha = (1/alpha) * log(sum(exp(alpha * (h*h))))
    
    alpha = 50.0 # Sharpness
    lr = 0.01
    steps = 1000
    
    # Time guard
    start_time = time.time()
    deadline = 100.0 # 100s limit
    
    for i in range(steps):
        if time.time() - start_time > deadline:
            break
            
        conv = np.convolve(h, h)
        
        # Softmax gradient
        # M = (1/alpha) log(sum(exp(alpha * conv)))
        # dM/dconv = exp(alpha * conv) / sum(exp(alpha * conv))
        # dconv/dh = [h, h, ..., h] (shifted)
        
        # Efficiently compute gradient of softmax conv
        # This is complex. 
        # Alternative: Just use the index of the max.
        # dM/dh = 2 * h_shifted where shift corresponds to max lag.
        
        max_idx = np.argmax(conv)
        
        # Gradient of (h*h) at max_idx wrt h
        # (h*h)_k = sum_i h_i h_{i+k}
        # d/dh_j (h*h)_k = h_{j+k} (if j+k < N) + h_{j-k} (if j-k >= 0)
        
        grad = np.zeros(N)
        # Contribution from positive lags
        # If max_idx is k, then h_i h_{i+k}
        # For a given h_j, it appears in h_j h_{j+k} (if j+k < N) and h_{j-k} h_j (if j-k >= 0)
        
        # Actually, d(h*h)_k / d(h) is the vector h shifted by -k and h shifted by k, summed?
        # Let's just use numerical gradient or a simpler heuristic.
        
        # Heuristic: Reduce the peak.
        # The peak is at max_idx.
        # To reduce the peak, we should reduce the correlation at that lag.
        # This suggests reducing h_i where h_i * h_{i+k} is large.
        
        # Simpler: Projected Gradient Descent on a proxy.
        # Let's just try random perturbations and keep the best (Simulated Annealing / Hill Climbing)
        # Given N=400, 120s is a lot of time for 400 dims.
        
        # Let's stick to the smooth max gradient if possible, or just return uniform if optimization is risky.
        # But uniform is 2.0. We know 1.25 is possible.
        
        # Let's try a specific pattern: "Triangle" window?
        # Or "Hamming" window?
        
        # Let's try a simple hill climbing with random steps.
        pass

    return h
