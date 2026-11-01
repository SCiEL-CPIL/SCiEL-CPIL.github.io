"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_score(f):
    """
    Computes the score for a given signed step function.
    Metric: Sum of cubes of absolute values of circular autocorrelation coefficients.
    This measures the 'flatness' of the autocorrelation. Lower is better.
    """
    fft_f = np.fft.fft(f)
    power_spectrum = fft_f * np.conj(fft_f)
    autocorr = np.fft.ifft(power_spectrum).real
    
    # Exclude DC term (k=0) which is constant = N
    score = np.sum(np.abs(autocorr[1:])**3)
    return score

def _local_search(f, max_iter=500, timeout=None):
    """
    Perform local search by flipping individual bits to improve score.
    """
    if timeout is not None:
        start_time = time.time()
    
    current = f.copy()
    current_score = _compute_score(current)
    improved = True
    
    while improved:
        if timeout is not None and (time.time() - start_time) > timeout:
            break
        improved = False
        # Try flipping each bit and see if it improves
        for i in range(N):
            if timeout is not None and (time.time() - start_time) > timeout:
                break
            original = current[i]
            current[i] = -original
            new_score = _compute_score(current)
            if new_score < current_score:
                current_score = new_score
                improved = True
            else:
                current[i] = original
    return current, current_score

def construct():
    """
    Construct a signed step function with N=400 equal intervals on [-1/4, 1/4].
    Uses simulated annealing with local search to minimize the L3 norm of the circular autocorrelation.
    """
    # Time budget: 110 seconds to leave buffer
    total_time = 110
    start_time = time.time()
    
    # Initialize with random signs
    current = np.random.choice([-1, 1], size=N)
    current_score = _compute_score(current)
    
    best = current.copy()
    best_score = current_score
    
    # Simulated annealing parameters
    T_start = 100.0
    T_end = 0.01
    cooling_rate = 0.995
    
    T = T_start
    no_improve_count = 0
    
    while (time.time() - start_time) < total_time:
        # Propose a move: flip k random bits
        k = np.random.choice([1, 2, 3, 5], p=[0.4, 0.3, 0.2, 0.1])
        flip_indices = np.random.choice(N, size=k, replace=False)
        
        candidate = current.copy()
        candidate[flip_indices] *= -1
        candidate_score = _compute_score(candidate)
        
        delta = candidate_score - current_score
        
        # Acceptance criterion
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            current = candidate
            current_score = candidate_score
            
            if current_score < best_score:
                best_score = current_score
                best = current.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Cool down
        T = max(T * cooling_rate, T_end)
        
        # Occasionally run local search
        if no_improve_count > 50 and (time.time() - start_time) < total_time - 20:
            local_result, local_score = _local_search(current, max_iter=200, timeout=10)
            if local_score < best_score:
                best_score = local_score
                best = local_result.copy()
                current = local_result.copy()
                current_score = local_score
            no_improve_count = 0
    
    # Final local search on best
    if (time.time() - start_time) < total_time - 5:
        final_result, final_score = _local_search(best, max_iter=1000, timeout=total_time - (time.time() - start_time) - 2)
        if final_score < best_score:
            best = final_result
            best_score = final_score
    
    return best
# EVOLVE-BLOCK-END
