"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _eval_norm_fast(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses FFT for dense sampling and golden section refinement for accuracy.
    Returns max |P(z)|.
    """
    n = len(coeffs)
    M = 8192
    # Pad coeffs to length M
    c_padded = np.zeros(M, dtype=complex)
    c_padded[:n] = coeffs
    # P(z_j) = sum c_k e^{2pi i j k / M}
    # This is M * IFFT(c_padded)_j
    P = M * np.fft.ifft(c_padded)
    mags = np.abs(P)
    
    # Find peaks
    # Get indices of top K peaks
    K = 30
    idx = np.argsort(mags)[-K:][::-1]
    
    max_val = 0.0
    theta = 2 * np.pi * np.arange(M) / M
    
    for i in idx:
        # Golden section refinement around theta_i
        w = 2 * np.pi / M
        a = theta[i] - w
        b = theta[i] + w
        
        # Golden section search for max of |P(z)|
        # f(theta) = |P(e^{i theta})|
        gr = (5 ** 0.5 - 1) / 2  # golden ratio
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        
        # Evaluate at c and d
        def f(theta_val):
            z_val = np.exp(1j * theta_val)
            # Horner's method
            p_val = 0.0
            for coeff in reversed(coeffs):
                p_val = p_val * z_val + coeff
            return np.abs(p_val)
        
        fc = f(c)
        fd = f(d)
        
        # Iterate
        for _ in range(40):
            if fc < fd:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = f(d)
            else:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = f(c)
        
        peak_val = max(fc, fd)
        if peak_val > max_val:
            max_val = peak_val
            
    return max_val

def _score_fast(coeffs):
    """
    Computes the score: max |P(z)| / sqrt(24)
    """
    n = len(coeffs)
    norm = _eval_norm_fast(coeffs)
    return norm / np.sqrt(n)

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the score.
    Uses a combination of spectral synthesis initialization, pair-flip local search,
    and simulated annealing.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 140  # 140s limit, leaving 10s buffer
    
    # Strategy 1: Spectral Synthesis Initialization
    # We want a sequence whose Fourier transform is as flat as possible.
    # Ideally, the DFT of the coefficients should be constant magnitude.
    # Let's try to construct a "flat" spectrum in frequency domain, then inverse FFT.
    # However, we need binary coefficients.
    # So, let's generate a random phase spectrum with flat magnitude, IFFT, and threshold to +/-1.
    
    def generate_spectral_candidate():
        # Flat magnitude spectrum
        # We can use a random phase for each frequency bin.
        M = 4096
        # We only care about the first N coefficients, so we need the IFFT to be real?
        # No, the coefficients are real (+/-1).
        # So the spectrum must be Hermitian symmetric.
        # H[k] = conj(H[M-k])
        
        # Generate random phases for k=0 to M/2
        phases = np.random.uniform(0, 2 * np.pi, M // 2 + 1)
        # Magnitude is 1 (flat)
        spectrum = np.ones(M // 2 + 1, dtype=complex) * np.exp(1j * phases)
        
        # Construct full Hermitian spectrum
        full_spectrum = np.zeros(M, dtype=complex)
        full_spectrum[:M // 2 + 1] = spectrum
        # For k > M/2, full_spectrum[k] = conj(full_spectrum[M-k])
        # Note: index 0 is DC, M/2 is Nyquist (if M is even)
        if M % 2 == 0:
            # Nyquist bin must be real
            full_spectrum[M // 2] = np.real(full_spectrum[M // 2])
            for k in range(M // 2 + 1, M):
                full_spectrum[k] = np.conj(full_spectrum[M - k])
        else:
            for k in range(M // 2 + 1, M):
                full_spectrum[k] = np.conj(full_spectrum[M - k])
        
        # Inverse FFT
        # IFFT gives time domain signal
        # x[n] = 1/M sum X[k] e^{2pi i n k / M}
        # We want x[0..N-1] to be real and +/- 1.
        x = np.fft.ifft(full_spectrum) * M
        x = np.real(x)
        
        # Threshold to +/- 1
        # The mean should be 0 ideally?
        # Let's just sign(x)
        # But we need to ensure it's a valid polynomial.
        # Sign might give 0 if x is 0.
        coeffs = np.sign(x[:N])
        # Replace 0 with 1
        coeffs[coeffs == 0] = 1
        
        return coeffs

    # Generate a few spectral candidates
    candidates = []
    for _ in range(10):
        candidates.append(generate_spectral_candidate())
    
    # Also include Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    candidates.append(np.array(p[:N], dtype=float))
    
    # Evaluate all candidates
    best_coeffs = None
    best_score = float('inf')
    
    for cand in candidates:
        s = _score_fast(cand)
        if s < best_score:
            best_score = s
            best_coeffs = cand.copy()
    
    # Local search with pair flips (inspiration from cell findings)
    # Single flip hill climbing gets stuck. Pair flips allow escaping local minima.
    
    def local_search_pair(coeffs, current_score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all pair flips
            for i in range(N):
                for j in range(i + 1, N):
                    coeffs[i] *= -1
                    coeffs[j] *= -1
                    new_score = _score_fast(coeffs)
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        # Revert
                        coeffs[i] *= -1
                        coeffs[j] *= -1
            # Also try single flips
            for i in range(N):
                coeffs[i] *= -1
                new_score = _score_fast(coeffs)
                if new_score < current_score:
                    current_score = new_score
                    improved = True
                else:
                    coeffs[i] *= -1
        
        return coeffs, current_score

    # Run local search on the best candidate
    best_coeffs, best_score = local_search_pair(best_coeffs, best_score, start_time + 60)
    
    # Simulated Annealing with pair flips
    T = 0.5
    cooling_rate = 0.995
    min_T = 1e-6
    
    # Precompute indices for pair flips
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    while T > min_T and time.time() < deadline:
        # Pick random move: 80% pair, 20% single
        if np.random.rand() < 0.8:
            i, j = pairs[np.random.randint(len(pairs))]
            best_coeffs[i] *= -1
            best_coeffs[j] *= -1
            new_score = _score_fast(best_coeffs)
            delta = new_score - best_score
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_score = new_score
            else:
                best_coeffs[i] *= -1
                best_coeffs[j] *= -1
        else:
            i = np.random.randint(N)
            best_coeffs[i] *= -1
            new_score = _score_fast(best_coeffs)
            delta = new_score - best_score
            
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                best_score = new_score
            else:
                best_coeffs[i] *= -1
        
        T *= cooling_rate
        
        # Periodic local search
        if time.time() % 10 < 1.0: # Roughly every 10s
            best_coeffs, best_score = local_search_pair(best_coeffs, best_score, time.time() + 2)
    
    # Final local search
    best_coeffs, best_score = local_search_pair(best_coeffs, best_score, deadline)
    
    # Return list of ints
    return [int(c) for c in best_coeffs]

# EVOLVE-BLOCK-END
