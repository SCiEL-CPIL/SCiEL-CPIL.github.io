import numpy as np
import time

N = 100
TIME_LIMIT = 110.0  # Safety margin

def _compute_max_overlap(h):
    """
    Computes the maximum overlap of h with shifted (1-h).
    Assumes circular shifts.
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N])
    """
    g = 1.0 - h
    max_ov = 0.0
    for k in range(N):
        shifted_g = np.roll(g, k)
        ov = np.dot(h, shifted_g)
        if ov > max_ov:
            max_ov = ov
    return max_ov

def construct():
    start_time = time.time()
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # If the analytical minimum is indeed the global minimum, this will return 1.0.
    # But we try to improve.
    
    # Use a simple local search with random perturbations
    # Constraint: sum(h) = N/2, h in [0,1]
    
    current_h = best_h.copy()
    current_score = best_score
    
    # Temperature for simulated annealing-like behavior
    # We don't know the scale, so we use relative improvement
    T = 0.01
    
    while time.time() - start_time < TIME_LIMIT:
        # Generate a random perturbation that preserves sum
        # Pick two random indices i, j
        i = np.random.randint(N)
        j = np.random.randint(N)
        if i == j:
            continue
            
        # Perturb amount
        delta = np.random.uniform(-0.1, 0.1)
        
        new_i = current_h[i] + delta
        new_j = current_h[j] - delta
        
        # Check bounds
        if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
            continue
            
        # Apply
        old_i = current_h[i]
        old_j = current_h[j]
        current_h[i] = new_i
        current_h[j] = new_j
        
        new_score = _compute_max_overlap(current_h)
        
        # Accept if better, or with probability exp(-diff/T) if worse
        if new_score < current_score:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = current_h.copy()
        else:
            # Metropolis criterion
            diff = new_score - current_score
            if diff < 0.001: # Only consider small worsenings
                prob = np.exp(-diff / T)
                if np.random.rand() < prob:
                    current_score = new_score
            # If worsened significantly, revert
            if current_score > best_score:
                # Revert if we got stuck in a bad local min? 
                # Actually, standard SA keeps the state.
                pass
        
        # Cool down
        T *= 0.999
        
        # If T is too small, restart with new random init?
        if T < 1e-4:
            # Random restart
            current_h = np.random.uniform(0, 1, N)
            # Project to sum N/2
            current_h = current_h - np.mean(current_h) + N/2.0/N
            # Clip
            current_h = np.clip(current_h, 0, 1)
            # Re-normalize sum? Clipping breaks sum.
            # Simple fix: scale
            s = np.sum(current_h)
            if s > 0:
                current_h = current_h * (N/2.0) / s
            # Clip again
            current_h = np.clip(current_h, 0, 1)
            current_score = _compute_max_overlap(current_h)
            T = 0.01

    return best_h
