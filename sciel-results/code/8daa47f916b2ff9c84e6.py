import numpy as np
import time

N = 400
DEADLINE = 120.0

def _compute_autocorr(f):
    """
    Compute the autocorrelation for lags 1 to N-1.
    R[t] = sum_{i=0}^{N-t-1} f[i] * f[i+t]
    """
    R = np.zeros(N)
    for t in range(1, N):
        R[t] = np.dot(f[:N-t], f[t:])
    return R

def construct():
    start_time = time.time()
    
    # Initialize with a random signed step function
    # Use a specific seed for reproducibility in the 'construct' call if needed, 
    # but usually we want to explore. 
    # Let's use a random start.
    np.random.seed(42)
    f = np.random.choice([-1, 1], size=N).astype(np.int8)
    
    # Initial autocorrelation
    R = _compute_autocorr(f)
    
    # We care about max absolute value for lags 1..N-1
    current_max = np.max(np.abs(R[1:]))
    
    best_f = f.copy()
    best_R = R.copy()
    best_max = current_max
    
    # Simulated Annealing / Hill Climbing
    # We want to minimize the max sidelobe.
    # We can use a temperature to accept worse solutions occasionally to escape local optima.
    
    max_iterations = 200000 # Should be plenty fast with O(N) updates
    T = 0.05 # Initial temperature
    cooling = 0.9999
    
    for i in range(max_iterations):
        if time.time() - start_time > DEADLINE - 2.0:
            break
        
        # Pick a random index to flip
        k = np.random.randint(N)
        
        # Calculate the change in R for all lags
        # R_new[t] = R[t] - 2 * f[k] * (f[k+t] + f[k-t])
        # Note: f[k+t] is valid only if k+t < N
        # f[k-t] is valid only if k-t >= 0
        
        # We create a delta array
        delta_R = np.zeros(N)
        
        # Lags t such that k+t < N => t < N-k
        max_t_plus = N - k
        if max_t_plus > 1:
            # t ranges from 1 to max_t_plus - 1
            # f[k+t] for t in 1..max_t_plus-1
            # Indices: k+1 to N-1
            # Corresponding t indices: 1 to max_t_plus-1
            delta_R[1:max_t_plus] -= 2 * f[k] * f[k+1:N]
            
        # Lags t such that k-t >= 0 => t <= k
        # t ranges from 1 to k
        if k >= 1:
            # f[k-t] for t in 1..k
            # Indices: k-1 down to 0
            # We need to map t to index in delta_R
            # delta_R[t] corresponds to lag t
            # f[k-t] is at index k-t
            # So for t in 1..k, we add -2 * f[k] * f[k-t]
            # Vectorized:
            # t indices: 1 to k
            # f indices: k-1 down to 0
            # delta_R[1:k+1] -= 2 * f[k] * f[k-1::-1]
            delta_R[1:k+1] -= 2 * f[k] * f[k-1::-1]
            
        R_new = R + delta_R
        new_max = np.max(np.abs(R_new[1:]))
        
        # Acceptance criterion
        if new_max < current_max:
            # Accept
            f[k] = -f[k]
            R = R_new
            current_max = new_max
            
            # Update best
            if new_max < best_max:
                best_max = new_max
                best_f = f.copy()
                best_R = R.copy()
        else:
            # Metropolis acceptance
            delta = new_max - current_max
            if delta > 0 and np.random.rand() < np.exp(-delta / T):
                f[k] = -f[k]
                R = R_new
                current_max = new_max
            else:
                # Reject, no change to f or R
                pass
                
        T *= cooling
        if T < 0.001:
            T = 0.001
            
    return best_f
