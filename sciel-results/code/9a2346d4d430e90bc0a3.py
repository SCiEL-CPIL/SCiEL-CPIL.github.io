import numpy as np
import time

N = 100

def _max_overlap(h):
    """Calculate the maximum overlap of h with shifted 1-h."""
    max_val = 0.0
    for k in range(N):
        # h_i * (1 - h_{i+k})
        # Using circular shift
        shifted_h = np.roll(h, k)
        overlap = np.sum(h * (1 - shifted_h))
        if overlap > max_val:
            max_val = overlap
    return max_val

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Uniform solution
    h_uniform = np.full(N, 0.5)
    best_h = h_uniform.copy()
    best_score = _max_overlap(best_h) # Should be 25.0
    
    # Try to improve using local search
    # Since we suspect 25 is the lower bound, we might not improve.
    # But let's try a few random perturbations just in case.
    
    np.random.seed(42)
    h = h_uniform.copy()
    
    for i in range(100): # Limit iterations
        if time.time() > deadline:
            break
            
        # Random perturbation
        mask = np.random.choice(N, 10, replace=False)
        h[mask] += np.random.uniform(-0.05, 0.05, 10)
        h = np.clip(h, 0, 1)
        
        # Project to sum constraint
        diff = N/2 - np.sum(h)
        h += diff / N
        
        # Clip again
        h = np.clip(h, 0, 1)
        
        # Re-project (simple iterative)
        for _ in range(10):
            diff = N/2 - np.sum(h)
            h += diff / N
            h = np.clip(h, 0, 1)
            
        if abs(np.sum(h) - N/2) > 1e-5:
            continue
            
        score = _max_overlap(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
            
    return best_h
