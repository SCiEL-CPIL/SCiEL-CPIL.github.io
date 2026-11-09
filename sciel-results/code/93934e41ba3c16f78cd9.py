import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    N = 24
    
    # Use FFT-friendly grid for fast evaluation
    M = 65536  # 2^16 points, FFT-friendly
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    Z = np.exp(1j * theta)
    
    def sup_norm_fft(coeffs):
        """Compute sup norm using FFT for speed."""
        # coeffs: array of shape (N,) with values +/- 1
        # Use FFT to evaluate P(z) at all M points
        # P(z) = sum c_k z^k, so we need DFT of coeffs
        vals = np.fft.fft(coeffs.astype(complex))[:M]
        return np.max(np.abs(vals))
    
    def refine_peak(coeffs, max_idx, M):
        """Refine the peak around max_idx using golden section."""
        # Find the actual theta at max_idx
        theta_max = 2 * np.pi * max_idx / M
        
        # Define f(theta) = |P(e^{i theta})|
        def f(t):
            z = np.exp(1j * t)
            # Horner's method or direct sum
            result = 0.0
            for k in range(N - 1, -1, -1):
                result = result * z + coeffs[k]
            return np.abs(result)
        
        # Golden section search in a small window around theta_max
        delta = 2 * np.pi / M
        a = theta_max - delta
        b = theta_max + delta
        
        # Golden ratio
        gr = (1 + np.sqrt(5)) / 2
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        
        fc = f(c)
        fd = f(d)
        
        for _ in range(50):  # 50 iterations for high accuracy
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - (b - a) / gr
                fc = f(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + (b - a) / gr
                fd = f(d)
        
        return max(fc, fd)
    
    def full_sup_norm(coeffs):
        """Compute sup norm with peak refinement."""
        # First get coarse estimate
        coarse_max = sup_norm_fft(coeffs)
        max_idx = np.argmax(np.abs(np.fft.fft(coeffs.astype(complex))[:M]))
        
        # Refine the peak
        refined = refine_peak(coeffs, max_idx, M)
        
        return refined
    
    # Generate self-reciprocal polynomials: c_k = c_{N-1-k}
    # This reduces 24 bits to 12 independent bits
    # Further, if we require P(z) = P(-z) (even polynomial), then c_k = c_{k+2} for even k
    # But let's start with just self-reciprocal: 12 free bits
    
    best_score = float('inf')
    best_coeffs = None
    
    # Exhaustive search over 12-bit self-reciprocal polynomials
    # c = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c11, c10, c9, c8, c7, c6, c5, c4, c3, c2, c1, c0]
    
    for mask in range(4096):  # 2^12 = 4096 combinations
        if time.time() > deadline:
            break
        
        # Extract 12 bits
        bits = [(mask >> i) & 1 for i in range(12)]
        c_first_half = np.array([1 if b == 1 else -1 for b in bits])
        
        # Build full polynomial (self-reciprocal)
        coeffs = np.concatenate([c_first_half, c_first_half[::-1]])
        
        # Evaluate
        score = sup_norm_fft(coeffs)
        
        if score < best_score:
            best_score = score
            best_coeffs = coeffs.copy()
    
    # If we have time, refine the best candidate
    if best_coeffs is not None and time.time() < deadline - 10:
        refined_score = full_sup_norm(best_coeffs)
        if refined_score < best_score:
            best_score = refined_score
    
    # If best_coeffs is None (shouldn't happen), return a default
    if best_coeffs is None:
        return [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1]
    
    return [int(x) for x in best_coeffs]
