import numpy as np
import time

N = 600

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Objective: Minimize max(conv(h, h)) / (sum(h) * max(h))
    # This metric is scale-invariant.
    # We can fix max(h) = 1 to simplify.
    # Then minimize max(conv(h, h)) / sum(h).
    # Subject to h >= 0, max(h) = 1.
    
    # Let's use a simple local search or gradient step.
    # Initialize with a Gaussian or Rectangle.
    
    best_score = 1.0
    best_h = np.ones(N)
    
    # Try a few parametric shapes
    # 1. Rectangle (Score 1)
    # 2. Triangle
    # 3. Gaussian
    
    def calc_score(h):
        if np.max(h) == 0:
            return float('inf')
        c = np.convolve(h, h, mode='full')
        max_c = np.max(c)
        sum_h = np.sum(h)
        max_h = np.max(h)
        return max_c / (sum_h * max_h)
    
    # Parametric: Triangle
    # h = 1 - |2*i/N - 1|
    h_tri = np.array([1 - abs(2*i/N - 1) for i in range(N)])
    score_tri = calc_score(h_tri)
    if score_tri < best_score:
        best_score = score_tri
        best_h = h_tri.copy()
        
    # Parametric: Gaussian
    sigma = N / 10.0
    x = np.arange(N)
    h_gauss = np.exp(-0.5 * ((x - N/2.0) / sigma) ** 2)
    h_gauss /= np.max(h_gauss)
    score_gauss = calc_score(h_gauss)
    if score_gauss < best_score:
        best_score = score_gauss
        best_h = h_gauss.copy()
        
    # Local Optimization using Gradient Descent (approximate)
    # Since max is not differentiable, we can use subgradient or smooth approximation.
    # Or just random perturbations.
    
    current_h = best_h.copy()
    current_score = best_score
    
    step_size = 0.01
    for step in range(2000):
        if time.time() > deadline:
            break
            
        # Random perturbation
        noise = np.random.normal(0, step_size, N)
        new_h = current_h + noise
        new_h = np.clip(new_h, 0, None)
        
        if np.max(new_h) == 0:
            continue
            
        new_h /= np.max(new_h) # Normalize max to 1
        
        new_score = calc_score(new_h)
        
        if new_score < current_score:
            current_score = new_score
            current_h = new_h
            if current_score < best_score:
                best_score = current_score
                best_h = current_h.copy()
            step_size *= 1.05 # Increase step if improving? Or keep constant
        else:
            step_size *= 0.95 # Decrease step
            
    return list(best_h)

# EVOLVE-BLOCK-END
