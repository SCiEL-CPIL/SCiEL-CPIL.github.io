"""
Third autocorrelation inequality: minimize the upper bound on C3.

The score is the raw value (lower is better). The interface returns a sequence of
signed step heights (400 values). We use simulated annealing with:
1. Multiple diverse initializations (Thue-Morse, random, chaotic)
2. Simulated annealing to escape local minima
3. Deadline guard to ensure we finish within time limits

The objective: for a signed step function f with N intervals, the autoconvolution
upper bound C3 is related to the maximum of the autocorrelation. We want to
minimize the maximum absolute autocorrelation value (excluding the trivial
self-correlation at lag 0).

Structural change: Instead of greedy coordinate descent (which gets stuck in local
minima), we use simulated annealing which can escape local minima by accepting
worse solutions with a temperature-dependent probability. We also try multiple
initialization strategies to explore different basins of attraction.
"""

import numpy as np
import time

N = 400

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n: t(i) = parity of popcount(i)."""
    return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.float64)

def _autocorr_max(x):
    """
    Compute the maximum absolute value of the autocorrelation of x,
    excluding lag 0.
    
    For a step function with heights x[0..N-1], the autocorrelation at lag k
    is sum_{i} x[i] * x[i+k] (for the discrete version).
    
    We use FFT-based computation for efficiency.
    """
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    
    X = np.fft.rfft(x, size)
    # Autocorrelation via power spectrum
    ac = np.fft.irfft(X * np.conj(X), size)[:n]
    
    # ac[0] is the self-correlation (sum of squares), skip it
    # We want max |ac[k]| for k >= 1
    if n <= 1:
        return 0.0
    
    return np.max(np.abs(ac[1:]))

def _chaotic_init(r, x0, n):
    """
    Generate a sequence using the logistic map: x_{k+1} = r * x_k * (1 - x_k)
    Then threshold at 0.5 to get ±1 values.
    """
    x = np.zeros(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i-1] * (1.0 - x[i-1])
    # Threshold: values > 0.5 become +1, else -1
    return np.where(x > 0.5, 1.0, -1.0)

def _simulated_annealing(x0, start_time, deadline, T0=1.0, T_min=1e-6, alpha=0.9995):
    """
    Run simulated annealing starting from x0.
    Returns the best solution found.
    """
    x = x0.copy()
    current_max = _autocorr_max(x)
    best_x = x.copy()
    best_max = current_max
    
    T = T0
    n = len(x)
    
    while time.time() < deadline - 5.0 and T > T_min:
        # Try a random flip
        idx = np.random.randint(0, n)
        old_val = x[idx]
        x[idx] = -old_val
        new_max = _autocorr_max(x)
        
        delta = new_max - current_max
        
        # Accept if better, or with probability exp(-delta/T) if worse
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-12)):
            current_max = new_max
            if new_max < best_max:
                best_max = new_max
                best_x = x.copy()
        else:
            # Reject: revert
            x[idx] = old_val
        
        # Cool down
        T *= alpha
    
    return best_x, best_max

def construct():
    """
    Construct a signed step function with 400 equal intervals that minimizes
    the autoconvolution upper bound C3.
    
    Strategy (structural change from greedy descent to simulated annealing):
    1. Try multiple diverse initializations:
       - Thue-Morse sequence
       - Random ±1 sequence
       - Chaotic logistic map sequences with different parameters
    2. For each initialization, run simulated annealing
    3. Return the best solution found across all runs
    4. Use deadline guard to ensure we finish within time limits
    """
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin (limit is 120s, kill at 150s)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Thue-Morse
    initializations.append(_thue_morse(N) * 2.0 - 1.0)
    
    # 2. Random
    initializations.append(np.random.choice([-1.0, 1.0], size=N))
    
    # 3. Chaotic logistic map with different parameters near edge of chaos
    # Edge of chaos for logistic map is around r = 3.57-4.0
    chaotic_params = [
        (3.5, 0.1),
        (3.57, 0.2),
        (3.6, 0.3),
        (3.7, 0.4),
        (3.8, 0.5),
        (3.9, 0.6),
        (4.0, 0.7),
        (3.99, 0.8),
    ]
    for r, x0 in chaotic_params:
        initializations.append(_chaotic_init(r, x0, N))
    
    # Run simulated annealing on each initialization
    best_overall_x = None
    best_overall_max = float('inf')
    
    # Allocate time budget across initializations
    # Give more time to the first few (Thue-Morse and random are often good)
    n_inits = len(initializations)
    time_per_init = (deadline - start_time - 5.0) / n_inits
    
    for i, init in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
        
        # Adjust temperature based on initialization quality
        init_max = _autocorr_max(init)
        
        # Higher initial T for worse initializations to escape local minima
        T0 = max(0.1, min(5.0, init_max * 0.5))
        
        # Run simulated annealing
        local_best_x, local_best_max = _simulated_annealing(
            init, start_time, deadline, T0=T0
        )
        
        # Update global best
        if local_best_max < best_overall_max:
            best_overall_max = local_best_max
            best_overall_x = local_best_x.copy()
    
    # Final polish: run greedy descent on the best solution found
    if best_overall_x is not None and time.time() < deadline - 10.0:
        x = best_overall_x.copy()
        current_max = _autocorr_max(x)
        
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            order = np.random.permutation(N)
            for idx in order:
                if time.time() > deadline - 5.0:
                    break
                old_val = x[idx]
                x[idx] = -old_val
                new_max = _autocorr_max(x)
                if new_max < current_max - 1e-12:
                    current_max = new_max
                    improved = True
                else:
                    x[idx] = old_val
        
        if current_max < best_overall_max:
            best_overall_max = current_max
            best_overall_x = x
    
    if best_overall_x is None:
        # Fallback: return Thue-Morse
        return _thue_morse(N) * 2.0 - 1.0
    
    return best_overall_x

# For testing
if __name__ == "__main__":
    x = construct()
    print(f"Length: {len(x)}")
    print(f"Unique values: {np.unique(x)}")
    print(f"Max autocorrelation (excl lag 0): {_autocorr_max(x)}")
