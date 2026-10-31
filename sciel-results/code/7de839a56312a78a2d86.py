"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_c3(f):
    """Compute the upper bound C3 from the step function heights.
    
    We interpret C3 as the maximum absolute linear autocorrelation over non-zero lags,
    normalized by the number of terms, to make it comparable across sequences.
    """
    # f is a 1D array of length N with values +1 or -1
    n = len(f)
    # Compute linear autocorrelation for lags 1 to n-1
    # autocorr[k] = sum_{i=0}^{n-1-k} f[i] * f[i+k]
    max_abs = 0.0
    for k in range(1, n):
        ac = np.dot(f[:n-k], f[k:])
        abs_ac = abs(ac)
        if abs_ac > max_abs:
            max_abs = abs_ac
    # Normalize by n to get a per-element average
    return max_abs / n

def _evaluate(f):
    """Return the C3 value (lower is better)."""
    return _compute_c3(f)

def construct():
    """Construct a signed step function (±1) with 400 intervals to minimize C3."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initialize with random ±1 sequence
    rng = np.random.RandomState(42)
    best_f = rng.choice([-1, 1], size=N)
    best_val = _evaluate(best_f)
    
    # Simulated annealing
    current_f = best_f.copy()
    current_val = best_val
    
    # Parameters
    T = 1.0  # Initial temperature
    cooling = 0.9999
    min_T = 0.01
    
    while time.time() < deadline and T > min_T:
        # Pick a random position to flip
        idx = rng.randint(N)
        old_val = current_f[idx]
        current_f[idx] = -old_val
        
        new_val = _evaluate(current_f)
        
        # Accept if better or with probability exp(-(new-old)/T)
        if new_val < current_val or (rng.random() < np.exp(-(new_val - current_val) / T)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_f = current_f.copy()
        else:
            current_f[idx] = old_val  # Revert
        
        T *= cooling
    
    return best_f
# EVOLVE-BLOCK-END
