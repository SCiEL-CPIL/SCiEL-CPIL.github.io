"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def compute_score(x):
    """Compute max|autoconv| / (sum x)^2 for sequence x."""
    # Autoconvolution (linear)
    autoconv = np.correlate(x, x, mode='full')
    max_abs_ac = np.max(np.abs(autoconv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_ac / (integral * integral)


def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Start with all-ones
    x = np.ones(N, dtype=np.float64)
    best_x = x.copy()
    best_score = compute_score(best_x)
    
    # Local search: flip individual bits
    # For each position, try flipping sign and see if score improves
    n_iter = 0
    while time.time() < deadline:
        n_iter += 1
        # Pick a random position to flip
        idx = np.random.randint(N)
        x[idx] = -x[idx]
        new_score = compute_score(x)
        
        if new_score < best_score:
            best_score = new_score
            best_x = x.copy()
        else:
            # Revert
            x[idx] = -x[idx]
        
        # Check deadline
        if time.time() > deadline - 5:
            break
    
    return best_x
# EVOLVE-BLOCK-END
