"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
def construct():
    # Deadline: 145 seconds (safety margin for 150s limit)
    start_time = time.time()
    time_limit = 145.0

    def compute_score(a):
        fa = np.fft.fft(a)
        power = fa * np.conj(fa)
        R = np.fft.ifft(power).real
        # R[0] is the sum of squares, which is N for +/-1 sequences.
        # We sum the squares of the non-zero lags.
        return np.sum(R[1:]**2)

    def compute_score_from_R(R):
        return np.sum(R[1:]**2)

    # Initialize with Thue-Morse sequence
    # Thue-Morse has excellent autocorrelation properties
    a = np.array([1.0 if bin(i).count('1') % 2 == 0 else -1.0 for i in range(N)])
    
    score = compute_score(a)
    best_a = a.copy()
    best_score = score

    # Efficient local search using FFT-based score updates
    # For flipping a single element a[i] -> -a[i], the change in autocorrelation can be computed
    # But for simplicity and robustness, we'll use direct FFT computation but optimize the loop
    
    # First, do a quick greedy pass to find a good local minimum
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for idx in range(N):
            if (time.time() - start_time) > time_limit:
                break
            
            a[idx] *= -1.0
            new_score = compute_score(a)
            
            if new_score < score:
                score = new_score
                improved = True
            else:
                a[idx] *= -1.0
                
    if score < best_score:
        best_score = score
        best_a = a.copy()

    # If time remains, try multiple random restarts with local optimization
    if (time.time() - start_time) < time_limit:
        rng = np.random.default_rng(42)
        max_restarts = 10
        restarts_done = 0
        
        for restart in range(max_restarts):
            if (time.time() - start_time) > time_limit:
                break
            
            # Generate a random initial sequence
            a = rng.choice([-1.0, 1.0], size=N)
            score = compute_score(a)
            
            # Quick greedy optimization
            improved = True
            while improved and (time.time() - start_time) < time_limit:
                improved = False
                for idx in range(N):
                    if (time.time() - start_time) > time_limit:
                        break
                    
                    a[idx] *= -1.0
                    new_score = compute_score(a)
                    
                    if new_score < score:
                        score = new_score
                        improved = True
                    else:
                        a[idx] *= -1.0
                        
            if score < best_score:
                best_score = score
                best_a = a.copy()
            
            restarts_done += 1

    return best_a
# EVOLVE-BLOCK-END
