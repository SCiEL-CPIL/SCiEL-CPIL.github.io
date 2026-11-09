import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 130.0  # safety margin below 150s limit
    
    # Choose N that balances autoconv magnitude vs computation time
    # Score = max_abs_autoconv / (sum(f))^2
    # For +/-1 sequence of length N: sum ~ sqrt(N) in magnitude (random) or controlled
    # autoconv peaks ~ N for random, but can be reduced
    
    # Strategy: Use m-sequence (maximal length LFSR sequence) as base
    # m-sequence of length 2^m - 1 has optimal periodic autocorrelation
    # For linear autocorrelation, we get good sidelobe suppression
    
    # Try different m values and pick best
    # m=12: N=4095, m=13: N=8191, m=14: N=16383, m=15: N=32767
    # These are manageable for FFT-based autoconvolution
    
    def lfsr_msequence(m, primitive_poly):
        """Generate m-sequence of length 2^m - 1 using LFSR.
        primitive_poly: list of tap positions (0-indexed from LSB)
        """
        period = (1 << m) - 1
        state = 1  # non-zero initial state
        seq = np.zeros(period, dtype=np.int8)
        for i in range(period):
            seq[i] = 1 if (state & 1) == 1 else -1
            # Feedback: XOR of bits at tap positions
            feedback = 0
            for tap in primitive_poly:
                feedback ^= (state >> tap) & 1
            state = (state >> 1) | (feedback << (m - 1))
        return seq
    
    def compute_score(f):
        """Compute max_abs_autoconv / (sum(f))^2"""
        n = len(f)
        s = np.sum(f)
        if abs(s) < 1e-10:
            return float('inf')
        # Autoconvolution via FFT
        nfft = 1
        while nfft < 2 * n:
            nfft <<= 1
        F = np.fft.rfft(f, n=nfft)
        A = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
        # Exclude lag 0 (index 0)
        max_abs = np.max(np.abs(A[1:]))
        return max_abs / (s * s)
    
    def local_search(f, max_iters, time_limit):
        """Local search: flip signs to reduce max absolute autoconv."""
        f = f.copy().astype(np.float32)
        n = len(f)
        
        def fast_autoconv_excl0(f):
            nfft = 1
            while nfft < 2 * n:
                nfft <<= 1
            F = np.fft.rfft(f, n=nfft)
            A = np.fft.irfft(F * np.conj(F), n=nfft)[:n]
            return A
        
        A = fast_autoconv_excl0(f)
        current_max = np.max(np.abs(A[1:]))
        
        # Greedy: find indices whose flip most reduces peak autoconv
        # For flip at index i, delta to autoconv at lag k is:
        # A'(k) = A(k) - 2*f[i]*sum(f[i+k] or f[i-k] depending on sign of k)
        # This is complex to compute exactly for all lags efficiently.
        # Instead, use random restarts with limited flips.
        
        best_f = f.copy()
        best_score = current_max / (np.sum(f) ** 2)
        
        for it in range(max_iters):
            if time.time() > time_limit:
                break
            # Try flipping a random index
            i = np.random.randint(n)
            f[i] *= -1
            
            A_new = fast_autoconv_excl0(f)
            new_max = np.max(np.abs(A_new[1:]))
            new_sum = np.sum(f)
            if abs(new_sum) > 1e-10:
                new_score = new_max / (new_sum ** 2)
                if new_score < best_score:
                    best_score = new_score
                    best_f = f.copy()
                else:
                    f[i] *= -1  # revert
            else:
                f[i] *= -1  # revert
        
        return best_f, best_score
    
    # Primitive polynomials for LFSR (x^m + x^k + 1 form)
    # taps are positions where x appears (excluding x^m)
    # For x^m + x^k + 1, taps = [0, k]
    primitive_polys = {
        4: [0, 1],      # x^4+x+1
        5: [0, 2],      # x^5+x^2+1
        6: [0, 1],      # x^6+x+1
        7: [0, 3],      # x^7+x^3+1
        8: [0, 4],      # x^8+x^4+1
        9: [0, 4],      # x^9+x^4+1
        10: [0, 3],     # x^10+x^3+1
        11: [0, 2],     # x^11+x^2+1
        12: [0, 6],     # x^12+x^6+1
        13: [0, 4],     # x^13+x^4+1
        14: [0, 5],     # x^14+x^5+1
        15: [0, 14],    # x^15+x^14+1
        16: [0, 5],     # x^16+x^5+1
    }
    
    best_overall_f = None
    best_overall_score = float('inf')
    
    # Try several m-sequence lengths
    for m in [12, 13, 14, 15]:
        if time.time() > deadline - 30:
            break
        if m not in primitive_polys:
            continue
        
        f = lfsr_msequence(m, primitive_polys[m])
        f = f.astype(np.float32)
        
        # Compute initial score
        s = np.sum(f)
        if abs(s) < 1e-10:
            # Adjust: flip one element to make sum non-zero
            f[0] *= -1
            s = np.sum(f)
        
        score = compute_score(f)
        
        # Local search refinement
        remaining_time = deadline - 15 - time.time()
        if remaining_time > 5:
            # Estimate how many iterations we can do
            # Each iteration involves an FFT of size ~2*N
            # For N=4095, FFT is fast (~microseconds)
            # Be conservative: 5000 iterations
            max_iters = min(5000, int(remaining_time * 200))
            f, score = local_search(f, max_iters, deadline - 15)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_f = f.copy()
    
    # If nothing worked, fallback
    if best_overall_f is None:
        # Use a simple random +/-1 sequence
        N = 4095
        best_overall_f = np.random.choice([-1.0, 1.0], size=N).astype(np.float32)
    
    # Final safety: ensure it's a valid numpy array
    return best_overall_f
# EVOLVE-BLOCK-END
