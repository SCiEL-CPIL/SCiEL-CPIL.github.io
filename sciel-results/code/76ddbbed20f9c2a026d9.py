import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # 1. Generate initial Golay-Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    c = np.array(p[:24], dtype=float)
    
    # 2. Define fast sup norm evaluation and peak tracking
    # M = 2^15 = 32768 points. For N=24, the grid is very fine.
    # We can precompute the DFT matrix or use FFT.
    # Using FFT is faster for single evaluations if we don't need arbitrary z.
    # However, for local search, we need to know WHERE the peaks are to guide flips.
    
    M = 2**15
    # Precompute the DFT matrix W[k, n] = exp(-2j pi k n / M) for n in 0..23
    # This allows O(M*N) update per flip if we do it naively, but FFT is O(M log M).
    # Let's stick to FFT for speed, but we need to identify top peaks.
    
    def get_supnorm_and_peaks(coeffs):
        c_padded = np.zeros(M, dtype=float)
        c_padded[:24] = coeffs
        F = np.fft.fft(c_padded)
        abs_F = np.abs(F)
        # Find the top K peaks to guide the search
        # We care about the global max, but sometimes reducing the second max helps
        # if the global max is stuck.
        # For now, just return the max value and the indices of the top 5 peaks.
        # The indices correspond to angles 2*pi*k/M.
        top_k = 10
        idx = np.argsort(abs_F)[::-1][:top_k]
        return float(np.max(abs_F)), idx, abs_F
    
    current_c = c.copy()
    current_val, current_peaks, _ = get_supnorm_and_peaks(current_c)
    best_c = current_c.copy()
    best_val = current_val
    
    # Local Search Strategy:
    # 1. Identify the global maximum peak index k_max.
    # 2. The value at k_max is sum c_n * exp(-2j pi k_max n / M).
    # 3. Flipping c_n changes the value by -2 * c_n * exp(-2j pi k_max n / M).
    # 4. We want to reduce |F(k_max)|.
    # 5. We can evaluate the new sup-norm for flips that target the current peak.
    
    # To be efficient, we don't need to recompute the whole FFT for every candidate flip
    # if we only care about the peak we are trying to reduce, BUT we must ensure
    # we don't create a larger peak elsewhere.
    # A hybrid approach:
    # - For each of the top 5 peaks, try flipping each of the 24 coefficients.
    # - For each candidate, compute the new value at the target peak.
    # - If it reduces the target peak significantly, do a full FFT check to see if
    #   the global sup-norm actually decreased.
    
    # Even better: Just do full FFT for the most promising candidates.
    # How to rank candidates?
    # For peak at index k, flipping coefficient n changes F[k] by delta = -2 * c[n] * W[k, n].
    # We want |F[k] + delta| < |F[k]|.
    
    # Precompute W for the 24 coefficients
    # W[k, n] = exp(-2j pi k n / M)
    # We only need this for the peaks we encounter.
    
    # Let's use a simpler, robust local search:
    # Iterate:
    # 1. Find global max peak index k_max.
    # 2. For each n in 0..23, calculate the change in F[k_max] if we flip c[n].
    # 3. Select the flip that most reduces |F[k_max]|.
    # 4. Apply the flip if it reduces the global sup-norm (verified by full FFT).
    # 5. If no single flip reduces the global sup-norm, try a 2-flip move?
    #    Or use simulated annealing with a very low temperature focused on the peak.
    
    # Given the small N=24, we can afford to be a bit more exhaustive in the local search.
    # Let's try a "Steepest Descent on Peaks" approach.
    
    # Optimization:
    # Precompute the vectors v_n = exp(-2j pi k n / M) for a given k.
    
    while time.time() < deadline:
        # Get current sup norm and the index of the maximum
        # We already have current_val, but we need the index.
        # Let's recompute peaks to be sure, it's cheap enough (32768 FFT).
        c_padded = np.zeros(M, dtype=float)
        c_padded[:24] = current_c
        F = np.fft.fft(c_padded)
        abs_F = np.abs(F)
        k_max = np.argmax(abs_F)
        current_val = abs_F[k_max]
        
        if current_val < best_val:
            best_val = current_val
            best_c = current_c.copy()
        
        # Try to reduce the peak at k_max
        # Compute the phasors for k_max
        # phase_n = -2 * pi * k_max * n / M
        n_range = np.arange(24)
        phasors = np.exp(-2j * np.pi * k_max * n_range / M)
        
        # Current value F[k_max]
        F_kmax = F[k_max]
        
        # For each n, if we flip c[n], the change is delta_n = -2 * c[n] * phasors[n]
        # New value = F_kmax + delta_n
        # We want to minimize |F_kmax + delta_n|
        
        deltas = -2 * current_c * phasors
        new_values = F_kmax + deltas
        new_mags = np.abs(new_values)
        
        # Find the flip that gives the smallest magnitude at k_max
        n_best = np.argmin(new_mags)
        min_mag_at_peak = new_mags[n_best]
        
        # It's possible that reducing this peak increases another peak more.
        # So we must verify with full FFT.
        # But we can first check if this flip is even promising.
        # If min_mag_at_peak >= current_val, then no single flip reduces the peak at k_max.
        # In that case, we might be stuck in a local minimum for single flips.
        
        if min_mag_at_peak < current_val:
            # Apply the flip
            current_c[n_best] *= -1
            
            # Verify global sup norm
            c_padded_new = np.zeros(M, dtype=float)
            c_padded_new[:24] = current_c
            F_new = np.fft.fft(c_padded_new)
            new_global_val = np.max(np.abs(F_new))
            
            # If the global sup norm actually increased (due to another peak),
            # we might still accept it if we are using SA, but for pure steepest descent
            # on the global max, we should reject it.
            # However, sometimes moving towards a local min of one peak is necessary.
            # Let's use a simple acceptance: accept if global max decreases.
            # If it doesn't, try the next best flip for this peak?
            
            if new_global_val < best_val:
                best_val = new_global_val
                best_c = current_c.copy()
            elif new_global_val > current_val:
                # Reject the move, try the next best candidate for reducing k_max
                # We need to find the next best n that reduces k_max and doesn't blow up global max.
                # This is getting complicated. Let's simplify:
                # Just accept if new_global_val < current_val.
                # If not, revert.
                current_c[n_best] *= -1
                # If we reverted, we are stuck.
                # Let's try a random move to escape local minimum.
                # Pick a random flip and accept if it improves global max.
                # Or just continue the loop, which will pick the same n_best again.
                # To avoid infinite loop on same move, we can add a small random perturbation
                # or try the second best n.
                
                # Let's try the second best n for reducing k_max
                # Sort n by new_mags
                sorted_n = np.argsort(new_mags)
                applied = False
                for n_candidate in sorted_n[1:5]: # Try next 4 candidates
                    if n_candidate == n_best: continue
                    current_c[n_candidate] *= -1
                    c_padded_check = np.zeros(M, dtype=float)
                    c_padded_check[:24] = current_c
                    F_check = np.fft.fft(c_padded_check)
                    val_check = np.max(np.abs(F_check))
                    if val_check < current_val:
                        # Accept
                        current_val = val_check
                        if current_val < best_val:
                            best_val = current_val
                            best_c = current_c.copy()
                        applied = True
                        break
                    else:
                        current_c[n_candidate] *= -1
                
                if not applied:
                    # Stuck. Do a random flip and accept if it improves.
                    n_rand = random.randint(0, 23)
                    current_c[n_rand] *= -1
                    c_padded_rand = np.zeros(M, dtype=float)
                    c_padded_rand[:24] = current_c
                    F_rand = np.fft.fft(c_padded_rand)
                    val_rand = np.max(np.abs(F_rand))
                    if val_rand < current_val:
                        current_val = val_rand
                        if current_val < best_val:
                            best_val = current_val
                            best_c = current_c.copy()
                    else:
                        current_c[n_rand] *= -1
        else:
            # No single flip reduces the peak at k_max.
            # We are in a local minimum for 1-flip moves.
            # Try a random 2-flip move? Or just random 1-flip.
            # Let's try a random 1-flip to escape.
            n_rand = random.randint(0, 23)
            current_c[n_rand] *= -1
            c_padded_rand = np.zeros(M, dtype=float)
            c_padded_rand[:24] = current_c
            F_rand = np.fft.fft(c_padded_rand)
            val_rand = np.max(np.abs(F_rand))
            if val_rand < current_val:
                current_val = val_rand
                if current_val < best_val:
                    best_val = current_val
                    best_c = current_c.copy()
            else:
                current_c[n_rand] *= -1

    return [int(x) for x in best_c]
