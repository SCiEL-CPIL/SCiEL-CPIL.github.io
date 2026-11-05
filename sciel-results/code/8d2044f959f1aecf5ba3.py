import numpy as np
import time

N = 400

def generate_thue_morse(n):
    """Generate Thue-Morse sequence of length n."""
    # Start with 0
    seq = np.array([0])
    while len(seq) < n:
        # Thue-Morse rule: t_{2n} = t_n, t_{2n+1} = 1 - t_n
        # This is equivalent to: new_seq = concat(seq, 1 - seq)
        seq = np.concatenate([seq, 1 - seq])
    return seq[:n]

def compute_autocorr_peak(x):
    """
    Compute the maximum absolute aperiodic autocorrelation 
    excluding lag 0.
    x is a numpy array of 1s and -1s.
    """
    # Full autocorrelation
    # np.correlate(x, x, 'full') returns array of length 2*N - 1
    # Index N-1 is lag 0.
    ac = np.correlate(x, x, mode='full')
    # We want max |ac[k]| for k != N-1
    # ac[0:N-1] are negative lags, ac[N:] are positive lags
    # Due to symmetry for real sequences, ac[k] == ac[2*N-2-k]
    # So we can just check one side, e.g., ac[N:] (positive lags)
    # But to be safe, check all except center.
    mask = np.ones(len(ac), dtype=bool)
    mask[N-1] = False
    return np.max(np.abs(ac[mask]))

def local_search(x, max_iters=10000, time_limit=None, start_time=None):
    """
    Local search using simulated annealing or hill climbing.
    """
    if start_time is None:
        start_time = time.time()
    
    x = x.copy()
    current_score = compute_autocorr_peak(x)
    best_x = x.copy()
    best_score = current_score
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    for i in range(max_iters):
        if time_limit and (time.time() - start_time) > time_limit:
            break
            
        # Pick a random index to flip
        idx = np.random.randint(N)
        
        # Flip
        x[idx] *= -1
        
        new_score = compute_autocorr_peak(x)
        delta = new_score - current_score
        
        if delta < 0:
            # Improvement
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_x = x.copy()
        else:
            # No improvement, accept with probability exp(-delta/T)
            if np.random.rand() < np.exp(-delta / T):
                current_score = new_score
            else:
                # Revert
                x[idx] *= -1
        
        T *= cooling_rate
        if T < min_T:
            T = min_T
            
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin for 120s limit
    
    # Initial guess: Thue-Morse
    x_tm = generate_thue_morse(N)
    x_tm = np.where(x_tm == 0, 1, -1) # Map 0->1, 1->-1
    
    best_x = x_tm
    best_score = compute_autocorr_peak(best_x)
    
    # Run multiple restarts of local search
    # Each restart starts from a perturbed Thue-Morse or the current best
    # We'll do a mix: start from Thue-Morse, then from best found, etc.
    
    restarts = 0
    while (time.time() - start_time) < deadline:
        # Perturb the initial Thue-Morse sequence for diversity in restarts
        # Or just restart from best_x with some random flips
        if restarts == 0:
            x_start = x_tm.copy()
        else:
            x_start = best_x.copy()
            # Add some random noise to escape local minima
            for _ in range(10):
                idx = np.random.randint(N)
                x_start[idx] *= -1
        
        # Run local search with a time slice
        # Estimate time per iteration to determine max_iters
        # Let's just run until time limit for this specific search instance
        # But we want multiple restarts, so let's give each restart a portion of time
        # Or just let the inner loop check the global deadline.
        
        x_res, score_res = local_search(x_start, max_iters=5000, time_limit=deadline, start_time=start_time)
        
        if score_res < best_score:
            best_score = score_res
            best_x = x_res
            # Reset time limit? No, we are bounded by global deadline.
            # If we found a better solution, we might want to polish it further.
            # The local search function already handles the time limit.
            
        restarts += 1
        if restarts > 50: # Safety check
            break

    return best_x

# Test run
if __name__ == "__main__":
    x = construct()
    print("Score:", compute_autocorr_peak(x))
