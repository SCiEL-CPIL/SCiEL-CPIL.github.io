"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s kill

    def max_autocorr(h):
        """
        Compute the maximum absolute autocorrelation at non-zero lags.
        For a sequence of length N, the autocorrelation at lag k is sum_{i=0}^{N-k-1} h[i]*h[i+k].
        We want to minimize the maximum of |autocorr[k]| for k=1..N-1.
        """
        # Use FFT to compute autocorrelation efficiently
        # Zero-pad to at least 2*N-1 to avoid circular convolution effects
        n_fft = 1 << (2 * N - 1).bit_length()
        F = np.fft.rfft(h, n=n_fft)
        power = F * np.conj(F)
        acf = np.fft.irfft(power, n=n_fft).real[:N]
        # acf[0] is the energy (sum of squares), which is N for ±1 sequences
        # acf[1:N] contains the autocorrelations for lags 1 to N-1
        # Due to numerical precision, we take the max absolute value
        return np.max(np.abs(acf[1:]))

    # Start with a random ±1 sequence
    rng = np.random.RandomState(42)
    best_h = rng.choice([1, -1], N).astype(np.float64)
    best_val = max_autocorr(best_h)

    def simulated_annealing(h, T_start, T_end, steps, deadline):
        """
        Run simulated annealing to minimize max autocorrelation.
        Returns (best_h, best_val).
        """
        current_h = h.copy()
        current_val = max_autocorr(current_h)
        local_best_h = current_h.copy()
        local_best_val = current_val
        
        # Pre-compute cooling factor
        cooling = (T_end / T_start) ** (1.0 / steps)
        T = T_start
        
        no_improve_count = 0
        stagnation_limit = 1000
        
        for step in range(steps):
            if time.time() > deadline:
                break
            
            # Pick random bit to flip
            i = rng.randint(N)
            current_h[i] = -current_h[i]
            
            new_val = max_autocorr(current_h)
            delta = new_val - current_val
            
            # Acceptance criterion
            if delta < 0 or (T > 0 and rng.random() < np.exp(-delta / T)):
                current_val = new_val
                if new_val < local_best_val:
                    local_best_val = new_val
                    local_best_h = current_h.copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                current_h[i] = -current_h[i]  # Revert
                no_improve_count += 1
            
            T *= cooling
            
            if no_improve_count > stagnation_limit:
                break
        
        return local_best_h, local_best_val

    # Multiple SA runs with different starting points and temperatures
    # We will run multiple SA runs, each starting from a random or best-so-far state
    run_count = 0
    max_runs = 20  # Limit number of full SA runs to ensure we finish in time
    
    while time.time() < deadline and run_count < max_runs:
        run_count += 1
        
        # Determine starting point: alternate between random and best-so-far
        if run_count == 1:
            h_start = best_h.copy()
        elif run_count % 2 == 0:
            h_start = best_h.copy()
        else:
            h_start = rng.choice([1, -1], N).astype(np.float64)
        
        # Adjust temperature based on current best value
        # Higher T for early runs to explore, lower T for later runs to exploit
        progress = run_count / max_runs
        T_start = 10.0 * (1.0 - progress) + 1.0
        T_end = 0.01
        
        # Number of steps per run: adapt based on remaining time
        time_per_run = (deadline - time.time()) / (max_runs - run_count + 1)
        # Estimate steps: assume ~50us per step (conservative)
        steps = int(time_per_run * 10000)
        steps = max(steps, 1000)  # Minimum 1000 steps
        steps = min(steps, 50000)  # Cap at 50k steps per run
        
        h_candidate, val_candidate = simulated_annealing(
            h_start, T_start, T_end, steps, deadline
        )
        
        if val_candidate < best_val:
            best_val = val_candidate
            best_h = h_candidate

    # If time remains, do a few more local search passes on the best solution
    if time.time() < deadline - 10.0:
        # Final refinement with very low temperature
        h_final, val_final = simulated_annealing(
            best_h.copy(), 0.1, 0.001, 10000, deadline
        )
        if val_final < best_val:
            best_val = val_final
            best_h = h_final

    return best_h
# EVOLVE-BLOCK-END
