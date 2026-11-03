import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    rng = np.random.default_rng(42)
    
    # Initialize with random signs
    x = rng.integers(0, 2, N) * 2 - 1
    
    def calc_cost(x):
        c = np.correlate(x, x, mode='full')
        pos_corr = c[N:]
        return np.sum(np.abs(pos_corr)**3)
        
    current_cost = calc_cost(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T_start = 5.0
    T_end = 0.001
    num_iterations = 200000  # ~2-4 seconds execution time
    
    for i in range(num_iterations):
        idx = rng.integers(0, N)
        
        old_val = x[idx]
        x[idx] = -old_val
        
        new_cost = calc_cost(x)
        delta = new_cost - current_cost
        
        accepted = False
        if delta < 0:
            accepted = True
        else:
            # Calculate temperature
            # Linear cooling
            T = T_start * (1.0 - i / num_iterations)
            if T > 0:
                # Check if we should accept the worse move
                # Use log to avoid underflow/overflow issues if needed, but delta/T is small usually
                if rng.random() < np.exp(-delta / T):
                    accepted = True
        
        if accepted:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = x.copy()
        else:
            # Revert the flip
            x[idx] = old_val
            
    return best_x
# EVOLVE-BLOCK-END
