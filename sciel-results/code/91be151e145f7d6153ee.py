import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    N_COEFFS = 24
    SQRT_N = math.sqrt(N_COEFFS)
    
    # FFT size for evaluation
    FFT_SIZE = 4096
    
    # Precompute roots of unity
    roots = np.exp(2j * np.pi * np.arange(FFT_SIZE) / FFT_SIZE)
    
    def evaluate_supnorm(coeffs):
        """Evaluate max|P(z)| over |z|=1 using FFT + refinement."""
        # FFT evaluation
        c = np.zeros(FFT_SIZE, dtype=np.complex128)
        c[:N_COEFFS] = coeffs
        P = np.fft.fft(c) / FFT_SIZE  # P(w_j) where w_j = exp(2*pi*i*j/FFT_SIZE)
        # Actually, we want P(z) = sum c_k z^k, so at z = w_j = exp(2*pi*i*j/N)
        # P(w_j) = sum c_k w_j^k = sum c_k exp(2*pi*i*j*k/N)
        # This is the DFT, but with the sign convention... let me check
        # np.fft.fft(c)[j] = sum_{k=0}^{N-1} c[k] * exp(-2*pi*i*j*k/N)
        # We want exp(+2*pi*i*j*k/N), so we need ifft or conjugate
        # P(w_j) = sum c_k exp(2*pi*i*j*k/N) = conj(sum c_k exp(-2*pi*i*j*k/N)) = conj(fft(c)[j])
        # But |P(w_j)| = |conj(fft(c)[j])| = |fft(c)[j]|
        # So we can just use abs(fft(c))
        P_vals = np.fft.fft(c)
        magnitudes = np.abs(P_vals)
        
        # Find top peaks for refinement
        # Get indices of top 5 peaks
        top_indices = np.argsort(magnitudes)[-5:][::-1]
        
        best_mag = 0
        for idx in top_indices:
            # Golden section refinement around idx/FFT_SIZE * 2*pi
            center_angle = 2 * math.pi * idx / FFT_SIZE
            # Search in a small window around the peak
            # The FFT bin width is 2*pi/FFT_SIZE, so search in [-bin_width, +bin_width]
            bin_width = 2 * math.pi / FFT_SIZE
            a = center_angle - bin_width
            b = center_angle + bin_width
            
            # Golden section search
            gr = (math.sqrt(5) - 1) / 2  # ~0.618
            c1 = b - gr * (b - a)
            c2 = a + gr * (b - a)
            
            def eval_at_angle(theta):
                z = np.exp(1j * theta)
                # P(z) = sum c_k z^k
                # Use Horner's method or direct sum
                val = 0
                zk = 1
                for k in range(N_COEFFS - 1, -1, -1):
                    val = val * z + coeffs[k]
                    # Actually, let's do it forward
                    pass
                # Simpler: direct computation
                powers = np.array([1.0])
                for k in range(1, N_COEFFS):
                    powers = np.append(powers, powers[-1] * z)
                return np.abs(np.sum(coeffs * powers))
            
            # Actually, let's use a more efficient evaluation
            def eval_at_angle_fast(theta):
                z = np.exp(1j * theta)
                # P(z) = sum_{k=0}^{23} c_k z^k
                # Use numpy
                k = np.arange(N_COEFFS)
                return np.abs(np.sum(coeffs * (z ** k)))
            
            f1 = eval_at_angle_fast(c1)
            f2 = eval_at_angle_fast(c2)
            
            for _ in range(30):  # 30 iterations for high accuracy
                if f1 < f2:
                    b = c2
                    c2 = c1
                    f2 = f1
                    c1 = b - gr * (b - a)
                    f1 = eval_at_angle_fast(c1)
                else:
                    a = c1
                    c1 = c2
                    f1 = f2
                    c2 = a + gr * (b - a)
                    f2 = eval_at_angle_fast(c2)
            
            peak_val = max(f1, f2)
            if peak_val > best_mag:
                best_mag = peak_val
        
        # Also check the overall max from FFT as a fallback
        fft_max = np.max(magnitudes)
        return max(best_mag, fft_max)
    
    def evaluate_supnorm_fast(coeffs):
        """Fast sup norm estimate using only FFT (no refinement)."""
        c = np.zeros(FFT_SIZE, dtype=np.complex128)
        c[:N_COEFFS] = coeffs
        P_vals = np.fft.fft(c)
        return np.max(np.abs(P_vals))
    
    # Initialize with Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N_COEFFS:
        p, q = p + q, p + [-c for c in q]
    best_coeffs = np.array(p[:N_COEFFS], dtype=np.float64)
    best_score = evaluate_supnorm(best_coeffs) / SQRT_N
    
    # Local search: flip individual coefficients
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        current_score = best_score
        
        for k in range(N_COEFFS):
            if time.time() > deadline:
                break
            
            # Try flipping coefficient k
            trial = best_coeffs.copy()
            trial[k] = -trial[k]
            
            # Fast evaluation first
            fast_score = evaluate_supnorm_fast(trial) / SQRT_N
            if fast_score < current_score * 0.999:  # Only do full eval if clearly better
                full_score = evaluate_supnorm(trial) / SQRT_N
                if full_score < current_score:
                    best_coeffs = trial
                    best_score = full_score
                    improved = True
    
    # Try pair flips to escape local minima
    if time.time() < deadline - 5:
        for _ in range(100):  # Limited number of pair flip attempts
            if time.time() > deadline:
                break
            
            # Pick two random distinct coefficients to flip
            i, j = np.random.choice(N_COEFFS, size=2, replace=False)
            trial = best_coeffs.copy()
            trial[i] = -trial[i]
            trial[j] = -trial[j]
            
            fast_score = evaluate_supnorm_fast(trial) / SQRT_N
            if fast_score < best_score * 0.999:
                full_score = evaluate_supnorm(trial) / SQRT_N
                if full_score < best_score:
                    best_coeffs = trial
                    best_score = full_score
    
    # Final polish: do another round of single flips with full evaluation
    if time.time() < deadline - 2:
        for k in range(N_COEFFS):
            if time.time() > deadline:
                break
            trial = best_coeffs.copy()
            trial[k] = -trial[k]
            full_score = evaluate_supnorm(trial) / SQRT_N
            if full_score < best_score:
                best_coeffs = trial
                best_score = full_score
    
    # Return as list of ints (+1 or -1)
    return [int(c) for c in best_coeffs]
