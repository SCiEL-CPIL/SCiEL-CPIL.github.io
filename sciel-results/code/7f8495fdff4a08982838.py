"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin before 150s kill

    def max_autocorr(h):
        """
        Compute the maximum absolute autocorrelation at non-zero lags.
        Optimized for speed using pre-allocated FFT buffer if possible, 
        but here we just use standard numpy FFT which is quite fast for N=400.
        """
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:N]
        return np.max(np.abs(acf[1:]))

    def simulated_annealing(h_init, max_iters, start_temp, end_temp, seed):
        """
        Perform Simulated Annealing on the discrete sequence.
        """
        rng = np.random.RandomState(seed)
        h = h_init.copy()
        current_val = max_autocorr(h)
        best_h = h.copy()
        best_val = current_val
        
        # Precompute indices for shuffling or just pick random index
        # We will pick a random index to flip each step
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update temperature (geometric cooling)
            if i < max_iters - 1:
                temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            else:
                temp = end_temp
            
            # Choose a random index to flip
            idx = rng.randint(N)
            old_val = h[idx]
            h[idx] = -old_val
            
            new_val = max_autocorr(h)
            delta = new_val - current_val
            
            # Accept if better or with probability exp(-delta/temp)
            if delta < 0 or rng.rand() < np.exp(-delta / temp):
                current_val = new_val
                if new_val < best_val:
                    best_val = new_val
                    best_h = h.copy()
            else:
                # Revert
                h[idx] = old_val
                
        return best_h, best_val

    rng = np.random.RandomState(42)
    
    best_h = None
    best_val = np.inf
    
    # Strategy: Multiple restarts of Simulated Annealing
    # Each restart uses a random initial sequence.
    num_restarts = 0
    total_time_budget = deadline - start_time - 5.0 # 5s safety margin
    
    while time.time() < deadline - 5.0:
        num_restarts += 1
        
        # Random initial sequence
        h_init = rng.choice([1, -1], N).astype(np.float64)
        
        # Parameters for SA
        # We don't know the exact optimal number of iterations, so we run for a fixed time chunk
        # or a large number of iterations, checking time inside the loop.
        # Let's estimate iterations per second. 
        # max_autocorr is ~ O(N log N). N=400. 
        # Let's try 5000 iterations per restart initially, and see how long it takes.
        # If it's too slow, we reduce. If too fast, we can increase restarts.
        
        # To be safe and adaptive:
        # Run SA for a target duration of ~2 seconds per restart.
        # Estimate speed: 
        # Let's just run a fixed large number and break on time.
        max_iters_sa = 10000
        start_temp = 1.0
        end_temp = 0.01
        
        h_res, val_res = simulated_annealing(h_init, max_iters_sa, start_temp, end_temp, seed=42 + num_restarts)
        
        if val_res < best_val:
            best_val = val_res
            best_h = h_res
        
        # If we've done many restarts, we might be converging. 
        # We can also try to refine the best found so far with a shorter, lower-temp SA.
        if num_restarts > 2 and best_h is not None:
             # Refine current best with lower temperature
             h_ref, val_ref = simulated_annealing(best_h, 5000, 0.1, 0.001, seed=100 + num_restarts)
             if val_ref < best_val:
                 best_val = val_ref
                 best_h = h_ref

    # Final fallback
    if best_h is None:
        best_h = rng.choice([1, -1], N).astype(np.float64)

    return best_h
# EVOLVE-BLOCK-END
