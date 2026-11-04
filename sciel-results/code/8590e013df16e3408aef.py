import numpy as np
import time

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # Initial guess: Sine wave (half cycle) or Random Smooth
    # Let's start with a "bump" which we suspect is good
    x = np.arange(N)
    h = 0.5 + 0.5 * np.sin(np.pi * x / N)
    # Normalize to sum N/2
    h = h - np.mean(h) + 0.5
    # Clip
    h = np.clip(h, 0, 1)
    # Re-normalize sum to N/2 via bisection on scale? 
    # Or just simple projection.
    
    def project(h):
        # Project to sum N/2 and [0,1]
        # Simple iterative projection
        for _ in range(10):
            h = np.clip(h, 0, 1)
            diff = np.sum(h) - N / 2
            h = h - diff / N
        # Final clip and adjust
        h = np.clip(h, 0, 1)
        # If sum is not exactly N/2, adjust the largest elements
        diff = N / 2 - np.sum(h)
        if diff > 0:
            # Add to elements with most space
            idx = np.argsort(h)
            for i in idx:
                if diff <= 0: break
                add = min(diff, 1 - h[i])
                h[i] += add
                diff -= add
        elif diff < 0:
            idx = np.argsort(-h)
            for i in idx:
                if diff >= 0: break
                sub = min(-diff, h[i])
                h[i] -= sub
                diff += sub
        return h

    h = project(h)
    
    best_h = h.copy()
    best_score = np.inf

    # Precompute shifts
    shifts = range(1, N)
    
    def get_score(h):
        # Calculate max overlap
        max_ov = 0
        for k in shifts:
            # C_k = sum h_i (1 - h_{i+k})
            # Only sum where i+k < N
            ov = np.sum(h[:N-k] * (1 - h[k:]))
            if ov > max_ov:
                max_ov = ov
        return max_ov

    current_score = get_score(h)
    if current_score < best_score:
        best_score = current_score
        best_h = h.copy()

    # Gradient Descent
    # We want to minimize max C_k.
    # Use soft-max or just pick the worst k.
    
    step_size = 0.05
    for iter in range(2000):
        if time.time() > deadline:
            break
            
        # Find worst shift
        worst_k = 0
        max_ov = -1
        for k in shifts:
            ov = np.sum(h[:N-k] * (1 - h[k:]))
            if ov > max_ov:
                max_ov = ov
                worst_k = k
        
        # Gradient of C_k with respect to h
        # C_k = sum_{i=0}^{N-k-1} h_i - h_i h_{i+k}
        # dC_k/dh_i:
        # Term 1: h_i appears in sum as +1 (if i < N-k)
        # Term 2: -h_i h_{i+k} -> -h_{i+k} (if i < N-k)
        #          -h_{i-k} h_i -> -h_{i-k} (if i >= k)
        
        grad = np.zeros(N)
        k = worst_k
        
        # Contribution from sum h_i
        grad[:N-k] += 1.0
        
        # Contribution from -h_i h_{i+k}
        # For i in 0..N-k-1:
        # d/dh_i (-h_i h_{i+k}) = -h_{i+k}
        # d/dh_{i+k} (-h_i h_{i+k}) = -h_i
        # So for index j:
        # If j < N-k: grad[j] += -h[j+k]
        # If j >= k: grad[j] += -h[j-k]
        
        grad[:N-k] -= h[k:]
        if k > 0:
            grad[k:] -= h[:N-k]
            
        # Update h
        h = h - step_size * grad
        
        # Project
        h = project(h)
        
        # Calculate score
        score = get_score(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    return best_h
