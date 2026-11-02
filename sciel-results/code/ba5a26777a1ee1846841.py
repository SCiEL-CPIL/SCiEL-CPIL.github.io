import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initialization
    # Use a random sequence as a starting point.
    # We could use a better init, but random + local search is very effective for N=400.
    # To be deterministic across runs if needed, but task doesn't require fixed seed.
    # However, for reproducibility in evaluation, a fixed seed is safer to avoid bad luck.
    rng = np.random.default_rng(42)
    s = rng.choice([-1, 1], size=N)
    
    # Compute initial autocorrelation
    # Linear autocorrelation: R[k] = sum_{i} s[i] s[i+k]
    # Using FFT:
    def compute_acf(s):
        n = len(s)
        fft_size = 1 << (2 * n - 1).bit_length()
        S = np.fft.rfft(s, fft_size)
        acf = np.fft.irfft(S * np.conj(S), fft_size)[:n]
        return acf

    acf = compute_acf(s)
    # Peak sidelobe: max of |acf[k]| for k != 0
    # Note: acf[0] is N.
    peak_sidelobe = np.max(np.abs(acf[1:]))
    best_peak = peak_sidelobe
    best_s = s.copy()
    
    # 2. Local Search (Greedy Hill Climbing)
    # We try to flip bits to reduce the peak sidelobe.
    # To speed up, we can evaluate the effect of flipping each bit.
    # Flipping s[i] changes acf[k] by -2 * s[i] * s[(i+k)%N] for circular?
    # No, it's linear autocorrelation.
    # R_new[k] = R_old[k] - 2 * s[i] * s[i+k] (if i+k < N) 
    # Actually, it's easier to just recompute or use incremental updates.
    # For N=400, recomputing FFT (size 800) is very fast (~10us).
    # 400 flips * 10us = 4ms per full pass.
    # We can do ~30,000 passes in 120s. This is overkill.
    
    # Optimization:
    # Instead of checking all 400 bits every time, we can check them in random order 
    # and accept the first improvement (First Improvement) or best improvement (Steepest Descent).
    # Steepest descent is slower per pass but fewer passes needed.
    # First improvement is faster per pass.
    
    # Let's use Steepest Descent: find the flip that minimizes the new peak sidelobe the most.
    # If no flip improves, stop or randomize.
    
    # To handle local minima, we can use a simple restart strategy or simulated annealing,
    # but given the time, simple greedy with random restarts if stuck is good.
    
    stuck_count = 0
    
    while time.time() < deadline:
        best_flip_idx = -1
        best_new_peak = best_peak
        best_new_s = None
        
        # Evaluate all possible flips
        for i in range(N):
            # Flip bit i
            old_val = s[i]
            s[i] = -old_val
            
            # Compute new ACF
            acf_new = compute_acf(s)
            peak_new = np.max(np.abs(acf_new[1:]))
            
            if peak_new < best_new_peak:
                best_new_peak = peak_new
                best_flip_idx = i
                best_new_s = s.copy()
            
            # Revert
            s[i] = old_val
        
        if best_flip_idx != -1:
            s = best_new_s
            best_peak = best_new_peak
            best_s = s.copy()
            stuck_count = 0
        else:
            # No improvement found in this pass
            stuck_count += 1
            if stuck_count > 100:
                # Random restart or perturbation
                # Perturb a few random bits
                num_perturb = 5
                indices = rng.choice(N, size=num_perturb, replace=False)
                s[indices] *= -1
                acf = compute_acf(s)
                peak_sidelobe = np.max(np.abs(acf[1:]))
                if peak_sidelobe < best_peak:
                    best_peak = peak_sidelobe
                    best_s = s.copy()
                stuck_count = 0
            elif stuck_count > 1000:
                # Full random restart
                s = rng.choice([-1, 1], size=N)
                acf = compute_acf(s)
                peak_sidelobe = np.max(np.abs(acf[1:]))
                if peak_sidelobe < best_peak:
                    best_peak = peak_sidelobe
                    best_s = s.copy()
                stuck_count = 0

    return best_s

