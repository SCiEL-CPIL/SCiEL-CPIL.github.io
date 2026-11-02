import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Simulated Annealing approach
    # This is distinct from the simple hill climbing in the seed.
    
    # Objective function
    def objective(x):
        # Calculate autocorrelation
        r = np.correlate(x, x, mode='full')
        max_abs_r = np.max(np.abs(r))
        s = np.sum(x)
        s_sq = s * s
        if s_sq == 0:
            return float('inf')
        return max_abs_r / s_sq

    # Initialize
    x = np.random.choice([-1, 1], size=N)
    current_score = objective(x)
    best_x = x.copy()
    best_score = current_score
    
    # SA Parameters
    T_start = 10.0
    T_end = 0.01
    alpha = 0.995  # Cooling rate
    num_iterations = 50000  # Adjust based on time limit
    
    # Time limit check
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    T = T_start
    
    for i in range(num_iterations):
        if time.time() - start_time > time_limit:
            break
            
        # Perturbation: Flip 1-3 random bits
        num_flips = np.random.randint(1, 4)
        indices = np.random.choice(N, num_flips, replace=False)
        x[indices] *= -1
        
        new_score = objective(x)
        
        # Acceptance criterion
        delta = new_score - current_score
        if delta < 0:
            # Better, always accept
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            # Worse, accept with probability exp(-delta/T)
            if np.random.random() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Reject, revert
                x[indices] *= -1
                
        # Cool
        T *= alpha
        if T < T_end:
            T = T_end

    return best_x
# EVOLVE-BLOCK-END
