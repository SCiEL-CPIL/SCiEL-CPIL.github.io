"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # 1. Initial Candidate: Truncated Golay-Rudin-Shapiro
    # This provides a good starting point with low sidelobes.
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:24], dtype=float)
    
    # 2. High-Resolution Evaluation Setup
    # We use a large FFT size to get a fine grid for sup-norm estimation.
    # N_fft must be a power of 2 for speed. 2^14 = 16384 points is dense enough
    # to capture peaks of a degree 23 polynomial reasonably well, and very fast.
    N_fft = 1 << 14
    n_coeffs = 24
    
    # Pre-compute the FFT of the basis? No, we just FFT the coefficients directly.
    # P(z) on unit circle is related to DFT.
    # To evaluate P(e^{i theta}) densely, we can zero-pad the coefficients to N_fft
    # and take the FFT. The result is P(e^{i 2pi k / N_fft}).
    
    def fast_score(coeffs):
        # coeffs: shape (24,)
        # Zero pad to N_fft
        # Note: np.fft.fft computes sum_{n=0}^{N-1} x_n e^{-i 2pi k n / N}
        # Our polynomial is sum c_k z^k. On unit circle z = e^{i theta}.
        # The values are P(e^{i 2pi k / N_fft}).
        # np.fft.fft(coeffs_padded) gives exactly these values (with sign convention check).
        # Actually, DFT definition: X_k = sum x_n e^{-i 2pi kn/N}.
        # We want sum c_n e^{i 2pi kn/N}.
        # This is equivalent to FFT of c_{-n} or taking conjugate of FFT of c_n (if c is real).
        # Since c is real (+/-1), P(e^{i theta}) = conj(P(e^{-i theta})).
        # |P(e^{i theta})| = |P(e^{-i theta})|.
        # So we can just take np.abs(np.fft.fft(coeffs_padded)).
        
        padded = np.zeros(N_fft, dtype=complex)
        padded[:n_coeffs] = coeffs
        vals = np.fft.fft(padded)
        # The peak might be between grid points, but for 24 coeffs and 16k points,
        # the grid error is negligible for ranking.
        # For the final score, we will refine.
        return np.max(np.abs(vals))
    
    def exact_score(coeffs):
        # Refine the peaks found by fast_score
        # Find the top few indices in the FFT output
        vals = np.abs(np.fft.fft(np.pad(coeffs, (0, N_fft - n_coeffs))))
        # Get indices of top 5 peaks
        top_indices = np.argsort(vals)[-5:]
        
        max_val = 0.0
        for idx in top_indices:
            # Golden section search around this peak
            # The peak is at angle theta = 2*pi*idx/N_fft
            # Search in [theta - pi/N_fft, theta + pi/N_fft]
            theta_center = 2 * np.pi * idx / N_fft
            width = 2 * np.pi / N_fft
            
            a = theta_center - width
            b = theta_center + width
            
            # Golden ratio
            gr = (math.sqrt(5) - 1) / 2
            
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            # Evaluate |P(e^{i theta})|
            def f(theta):
                z = np.exp(1j * theta)
                # Horner's rule for speed? Or just dot product with powers.
                # Powers: z^0, z^1, ..., z^23
                # We can compute powers iteratively or use z**k.
                # For 24 terms, direct sum is fast enough.
                # coeffs is real.
                # P(z) = sum c_k z^k
                # We can use np.vander or just a loop.
                # Let's use a loop for clarity and sufficient speed in this limited context.
                val = 0.0
                z_pow = 1.0 + 0.0j
                for k in range(24):
                    val += coeffs[k] * z_pow
                    z_pow *= z
                return abs(val)
            
            fc = f(c)
            fd = f(d)
            
            for _ in range(50): # 50 iterations is enough for 1e-9 accuracy
                if fc < fd:
                    b = d
                    d = c
                    fd = fc
                    c = b - gr * (b - a)
                    fc = f(c)
                else:
                    a = c
                    c = d
                    fc = fd
                    d = a + gr * (b - a)
                    fd = f(d)
            
            peak_val = max(fc, fd)
            if peak_val > max_val:
                max_val = peak_val
                
        return max_val / math.sqrt(24)

    best_val = exact_score(best_coeffs)
    current_coeffs = best_coeffs.copy()
    current_val = best_val
    
    # 3. Deterministic "Repulsion" + Local Search
    # Strategy: 
    # 1. Try flipping each coefficient one by one. If it reduces the *fast* score, keep it.
    # 2. If stuck, try pairs of flips.
    # 3. Use the exact score for final verification and comparison.
    
    # We will use the fast score for the main loop to maximize iterations.
    # We will periodically check the exact score to ensure we are not misled by grid artifacts.
    
    current_fast_val = fast_score(current_coeffs)
    
    # Phase 1: Single flips (Greedy)
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(24):
            # Flip i
            trial_coeffs = current_coeffs.copy()
            trial_coeffs[i] = -trial_coeffs[i]
            
            trial_fast_val = fast_score(trial_coeffs)
            
            if trial_fast_val < current_fast_val:
                # Accept
                current_coeffs = trial_coeffs
                current_fast_val = trial_fast_val
                improved = True
                
                # Check exact score occasionally
                if improved:
                    exact_v = exact_score(current_coeffs)
                    if exact_v < best_val:
                        best_val = exact_v
                        best_coeffs = current_coeffs.copy()
    
    # Phase 2: Pair flips (if time permits)
    if time.time() < deadline - 5:
        # Try swapping two coefficients
        for i in range(24):
            if time.time() > deadline - 5: break
            for j in range(i+1, 24):
                trial_coeffs = current_coeffs.copy()
                trial_coeffs[i] = -trial_coeffs[i]
                trial_coeffs[j] = -trial_coeffs[j]
                
                trial_fast_val = fast_score(trial_coeffs)
                
                if trial_fast_val < current_fast_val:
                    current_coeffs = trial_coeffs
                    current_fast_val = trial_fast_val
                    
                    exact_v = exact_score(current_coeffs)
                    if exact_v < best_val:
                        best_val = exact_v
                        best_coeffs = current_coeffs.copy()
    
    # Phase 3: Final local search with exact score (very limited steps)
    # If we have time, do a few steps of exact-score guided search
    if time.time() < deadline - 2:
        for _ in range(50):
            if time.time() > deadline - 2: break
            # Pick a random index to flip
            i = random.randint(0, 23)
            trial_coeffs = current_coeffs.copy()
            trial_coeffs[i] = -trial_coeffs[i]
            
            trial_exact_val = exact_score(trial_coeffs)
            
            if trial_exact_val < best_val:
                best_val = trial_exact_val
                best_coeffs = trial_coeffs.copy()
                current_coeffs = trial_coeffs.copy()
            else:
                # Don't update current_coeffs if it's worse, to stay near the best
                pass

    return best_coeffs.astype(int).tolist()
# EVOLVE-BLOCK-END
