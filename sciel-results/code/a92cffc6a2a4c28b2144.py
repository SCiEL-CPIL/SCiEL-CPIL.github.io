import numpy as np
import time

N = 400

def _compute_c3_max(f):
    """
    Compute max absolute cubic autocorrelation.
    C3(tau) = sum f[i] f[i+tau] f[i+2*tau]
    """
    max_val = 0.0
    # tau from 1 to N-2
    # Vectorized sum for each tau
    for tau in range(1, N - 1):
        # Indices: i, i+tau, i+2*tau
        # i ranges from 0 to N - 2*tau - 1
        length = N - 2 * tau
        if length <= 0:
            break
        # f[0:length] * f[tau:tau+length] * f[2*tau:2*tau+length]
        c3_val = np.sum(f[:length] * f[tau:tau+length] * f[2*tau:2*tau+length])
        abs_val = abs(c3_val)
        if abs_val > max_val:
            max_val = abs_val
    return max_val

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    # Thue-Morse is 0, 1, 1, 0, 1, 0, 0, 1 ...
    # Map 0->1, 1->-1 for signed sequence
    seq = np.zeros(n, dtype=int)
    for i in range(n):
        # Count bits
        # t(i) = parity of number of 1s in binary representation of i
        seq[i] = 1 if bin(i).count('1') % 2 == 0 else -1
    return seq

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Initial guess: Thue-Morse
    best_f = _thue_morse(N)
    best_score = _compute_c3_max(best_f)
    
    # Local Search / Simulated Annealing
    # We will use a hybrid: Greedy local search first, then SA.
    
    f = best_f.copy()
    current_score = best_score
    
    T = 10.0
    cooling = 0.99
    min_T = 0.1
    
    # Greedy Phase
    improved = True
    while improved and (time.time() < deadline - 30):
        improved = False
        # Try flipping each index
        for i in range(N):
            if time.time() > deadline - 30:
                break
            f[i] = -f[i]
            new_score = _compute_c3_max(f)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                f[i] = -f[i] # revert
            
            if current_score < best_score:
                best_score = current_score
                best_f = f.copy()
                
    # Simulated Annealing Phase
    T = 5.0
    while T > min_T and (time.time() < deadline - 5):
        # Random flip
        i = np.random.randint(N)
        f[i] = -f[i]
        new_score = _compute_c3_max(f)
        delta = new_score - current_score
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_f = f.copy()
        else:
            f[i] = -f[i] # revert
            
        T *= cooling
        
    return best_f
