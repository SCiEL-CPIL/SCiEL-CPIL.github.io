import numpy as np
import time

def _score(coeffs, nfft=16384):
    """Compute max |P(e^{iθ})| / sqrt(24) via dense FFT sampling."""
    # coeffs is a list/array of 24 values in {+1, -1}
    # P(z) = sum c_k z^k, z on unit circle
    # This is the DFT of the coefficient sequence
    c = np.array(coeffs, dtype=np.float64)
    # FFT gives DFT at frequencies 0, 1/N, ..., (N-1)/N
    # We want max over all θ, so use a large N
    spectrum = np.fft.fft(c, n=nfft)
    mag = np.abs(spectrum)
    return np.max(mag) / np.sqrt(24)

def _golden_refine(coeffs, theta0, width=0.02):
    """Refine the peak near theta0 using golden-section search."""
    c = np.array(coeffs, dtype=np.float64)
    def f(theta):
        z = np.exp(1j * theta)
        # P(z) = sum c_k z^k
        # Use Horner's method for accuracy
        val = 0.0 + 0.0j
        for k in range(23, -1, -1):
            val = val * z + c[k]
        return abs(val)
    
    # Golden section search in [theta0 - width, theta0 + width]
    gr = (np.sqrt(5.0) - 1.0) / 2.0
    a = theta0 - width
    b = theta0 + width
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = f(c)
    fd = f(d)
    for _ in range(60):
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
    return f((a + b) / 2.0) / np.sqrt(24)

def _evaluate(coeffs):
    """Full evaluation: dense FFT + golden-section refinement of top peaks."""
    c = np.array(coeffs, dtype=np.float64)
    nfft = 16384
    spectrum = np.fft.fft(c, n=nfft)
    mag = np.abs(spectrum)
    # Find top few peaks
    top_indices = np.argsort(mag)[-5:]
    best = 0.0
    for idx in top_indices:
        theta = 2.0 * np.pi * idx / nfft
        val = _golden_refine(coeffs, theta, width=np.pi / nfft * 2)
        if val > best:
            best = val
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Start from Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < 24:
        p, q = p + q, p + [-c for c in q]
    current = p[:24]
    current_score = _score(current)
    
    best = list(current)
    best_score = current_score
    
    # Hill climbing with random restarts
    rng = np.random.default_rng(42)
    
    def try_flip(coeffs, score, deadline):
        """Try flipping each coefficient, accept best improvement."""
        improved = False
        best_local = list(coeffs)
        best_local_score = score
        for i in range(24):
            if time.time() > deadline:
                break
            new_coeffs = list(coeffs)
            new_coeffs[i] = -new_coeffs[i]
            s = _score(new_coeffs)
            if s < best_local_score:
                best_local = new_coeffs
                best_local_score = s
                improved = True
        return best_local, best_local_score, improved
    
    # Main search loop
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        # Hill climb from current
        current, current_score, improved = try_flip(current, current_score, deadline)
        
        if current_score < best_score:
            best = list(current)
            best_score = current_score
        
        if not improved:
            # Local optimum reached, try random perturbation
            if time.time() > deadline:
                break
            # Perturb: flip 1-3 random coefficients
            n_flips = rng.integers(1, 4)
            current = list(best)
            for _ in range(n_flips):
                idx = rng.integers(0, 24)
                current[idx] = -current[idx]
            current_score = _score(current)
    
    # Final polish: try all single flips on best
    for i in range(24):
        if time.time() > deadline:
            break
        new_coeffs = list(best)
        new_coeffs[i] = -new_coeffs[i]
        s = _score(new_coeffs)
        if s < best_score:
            best = new_coeffs
            best_score = s
    
    # Also try pairs of flips on best if time permits
    if time.time() < deadline - 5.0:
        for i in range(24):
            if time.time() > deadline - 2.0:
                break
            for j in range(i+1, 24):
                if time.time() > deadline - 1.0:
                    break
                new_coeffs = list(best)
                new_coeffs[i] = -new_coeffs[i]
                new_coeffs[j] = -new_coeffs[j]
                s = _score(new_coeffs)
                if s < best_score:
                    best = new_coeffs
                    best_score = s
    
    return best
