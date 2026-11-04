"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s kill
    
    # Initialize with random +/- 1
    # Using a fixed seed for reproducibility in this block, but random is better for search
    # However, we can just use one random start and optimize.
    rng = np.random.RandomState(42)
    h = rng.choice([-1, 1], size=N).astype(np.float64)
    
    def get_cost(h):
        # Compute aperiodic autocorrelation
        # c[k] = sum_{i} h[i] h[i+k]
        # np.correlate(h, h, 'full') gives c[N-1] ... c[0] ... c[N-1]
        c = np.correlate(h, h, mode='full')
        return np.sum(c**2)
        
    current_cost = get_cost(h)
    best_cost = current_cost
    best_h = h.copy()
    
    # Local search: flip bits
    # We can try flipping each bit in a random order
    while time.time() < deadline:
        # Pick a random index to flip
        i = rng.randint(N)
        old_val = h[i]
        h[i] = -old_val
        
        new_cost = get_cost(h)
        
        if new_cost < current_cost:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_h = h.copy()
        else:
            h[i] = old_val
            
        # Occasionally restart? No, hill climbing on this landscape usually works well enough for a significant improvement over ones.
        # To escape local minima, we could add a small probability of accepting worse moves (Simulated Annealing),
        # but simple hill climbing is fast and robust.
        
    return best_h
# EVOLVE-BLOCK-END
