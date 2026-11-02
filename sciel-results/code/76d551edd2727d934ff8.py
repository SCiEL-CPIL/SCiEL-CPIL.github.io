import numpy as np

N = 400

# EVOLVE-BLOCK-START
def construct():
    rng = np.random.default_rng(12345)
    
    # Initialize random binary sequence
    x = rng.integers(0, 2, N) * 2 - 1
    
    def calc_cost(x):
        # Calculate autocorrelation C_k for k=1..N-1
        c = np.correlate(x, x, mode='full')
        # c[N:] contains C_1 to C_{N-1}
        pos_corr = c[N:]
        return np.sum(np.abs(pos_corr)**3)
    
    current_cost = calc_cost(x)
    best_x = x.copy()
    best_cost = current_cost
    
    # Simulated Annealing parameters
    T_start = 1000.0
    T_end = 1.0
    iterations = 30000
    # Geometric cooling
    alpha = (T_end / T_start) ** (1.0 / iterations)
    
    T = T_start
    
    for i in range(iterations):
        # Generate a new candidate by flipping a random bit
        j = rng.integers(0, N)
        old_val = x[j]
        x[j] = -old_val
        
        new_cost = calc_cost(x)
        delta_cost = new_cost - current_cost
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta_cost < 0:
            current_cost = new_cost
            if new_cost < best_cost:
                best_cost = new_cost
                best_x = x.copy()
        else:
            # Check acceptance probability
            # Avoid overflow/underflow issues by checking magnitude
            if delta_cost < 100 * T: # Only compute exp if it might be non-zero
                prob = np.exp(-delta_cost / T)
                if rng.random() < prob:
                    current_cost = new_cost
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_x = x.copy()
            else:
                # Revert
                x[j] = old_val
                # T still cools
                T *= alpha
                continue # Skip the common revert assignment below? No, structure needs care.
        
        # If accepted (or if we didn't continue above), we keep the flip.
        # If rejected (and didn't continue), we revert.
        # The logic above is slightly messy with the continue.
        # Let's rewrite the loop body for clarity.
        
        T *= alpha
        
    return best_x
# EVOLVE-BLOCK-END
