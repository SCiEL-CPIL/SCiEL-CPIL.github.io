"""Improved program: LFSR m-sequence generation with local search polish."""
N = 400
import numpy as np
import time

def compute_autocorrelation_bound(f):
    """Compute an upper bound on C3 based on autocorrelation sidelobes."""
    nfft = 1 << (2 * N - 1).bit_length()
    F = np.fft.rfft(f, n=nfft)
    acf = np.fft.irfft(F * np.conj(F), n=nfft)[:N]
    if abs(acf[0]) > 1e-12:
        sidelobe_energy = np.sum(acf[1:] ** 2)
        peak_energy = acf[0] ** 2
        bound = np.sqrt(sidelobe_energy / peak_energy)
    else:
        bound = np.sqrt(np.sum(acf[1:] ** 2))
    return bound

def construct():
    deadline = time.time() + 120.0
    safety_margin = 10.0  # seconds
    
    def lfsr_sequence(degree, poly_taps, length):
        """Generate a maximal-length sequence using Fibonacci LFSR."""
        state = [1] * degree
        seq = []
        # Precompute which state indices contribute to feedback
        fb_indices = []
        for tap in poly_taps:
            if tap == degree:
                continue
            if tap == 0:
                fb_indices.append(0)
            else:
                fb_indices.append(tap - 1)
        
        for _ in range(length):
            seq.append(state[-1])
            fb = 0
            for idx in fb_indices:
                fb ^= state[idx]
            state = state[1:] + [fb]
        return seq
    
    def bits_to_signs(bits, n):
        """Convert bit sequence to signs in {-1, 1}."""
        return np.array([1.0 if b == 1 else -1.0 for b in bits[:n]])
    
    # Known primitive polynomials for degree 9
    # Format: list of exponents including the leading term
    primitive_polys_deg9 = [
        [9, 4, 0],      # x^9 + x^4 + 1
        [9, 5, 3, 2, 0], # x^9 + x^5 + x^3 + x^2 + 1
        [9, 6, 3, 2, 0], # x^9 + x^6 + x^3 + x^2 + 1
        [9, 7, 5, 4, 0], # x^9 + x^7 + x^5 + x^4 + 1
        [9, 8, 5, 4, 0], # x^9 + x^8 + x^5 + x^4 + 1
    ]
    
    # Also try degree 8 (length 255) - won't work for 400, skip
    # Degree 10 gives length 1023, which is > 400
    primitive_polys_deg10 = [
        [10, 7, 0],      # x^10 + x^7 + 1
        [10, 8, 5, 4, 0], # x^10 + x^8 + x^5 + x^4 + 1
        [10, 9, 6, 5, 0], # x^10 + x^9 + x^6 + x^5 + 1
    ]
    
    candidates = []
    
    # Generate m-sequences from degree 9 polynomials
    for taps in primitive_polys_deg9:
        seq = lfsr_sequence(9, taps, 511)
        f = bits_to_signs(seq, N)
        candidates.append(f)
        # Sign-flipped version
        candidates.append(-f)
    
    # Generate m-sequences from degree 10 polynomials
    for taps in primitive_polys_deg10:
        seq = lfsr_sequence(10, taps, 1023)
        f = bits_to_signs(seq, N)
        candidates.append(f)
        candidates.append(-f)
    
    # Evaluate all candidates
    best_f = None
    best_bound = float('inf')
    
    for f in candidates:
        bound = compute_autocorrelation_bound(f)
        if bound < best_bound:
            best_bound = bound
            best_f = f.copy()
    
    # Local search: try flipping individual bits to improve the bound
    # Use a greedy approach with time budget
    if time.time() + safety_margin < deadline:
        f = best_f.copy()
        current_bound = best_bound
        improved = True
        max_rounds = 50
        
        while improved and time.time() + safety_margin < deadline:
            improved = False
            for round_num in range(max_rounds):
                if time.time() + safety_margin >= deadline:
                    break
                
                # Try flipping each bit and see if it improves
                for i in range(N):
                    if time.time() + safety_margin >= deadline:
                        break
                    
                    f[i] = -f[i]  # flip
                    new_bound = compute_autocorrelation_bound(f)
                    if new_bound < current_bound - 1e-12:
                        current_bound = new_bound
                        improved = True
                    else:
                        f[i] = -f[i]  # revert
                if not improved:
                    break
        
        best_f = f
        best_bound = current_bound
    
    return best_f
