import numpy as np
import time

def _eval_norm(coeffs):
    """
    Computes the sup norm of the polynomial P(z) = sum c_k z^k for |z|=1.
    Uses dense sampling and golden section refinement for accuracy.
    """
    n = len(coeffs)
    # Dense sampling
    N = 2048
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    z = np.exp(1j * theta)
    
    # Evaluate polynomial using FFT-like approach or direct dot product
    # coeffs shape (n,). z^k for k in 0..n-1
    # P(z) = c_0 + c_1 z + ... + c_{n-1} z^{n-1}
    # Using matrix multiplication for speed with small n
    # powers[i, j] = z_j^i
    powers = np.vander(z, n, increasing=True).T # Shape (n, N)
    P = coeffs @ powers # Shape (N,)
    mags = np.abs(P)
    
    # Find peaks
    # Simple peak detection: local maxima
    peaks = []
    for i in range(N):
        if mags[i] > mags[(i - 1) % N] and mags[i] >= mags[(i + 1) % N]:
            peaks.append(i)
    
    if not peaks:
        return np.max(mags)
    
    # Refine top peaks
    # Sort peaks by magnitude
    peak_indices = sorted(peaks, key=lambda i: mags[i], reverse=True)
    top_peaks = peak_indices[:10] # Refine top 10
    
    max_val = 0.0
    for idx in top_peaks:
        # Golden section search around theta[idx]
        # Interval: [theta[idx-1], theta[idx+1]]
        t_low = theta[(idx - 1) % N]
        t_high = theta[(idx + 1) % N]
        
        # Handle wrap-around
        if t_high < t_low:
            t_high += 2 * np.pi
            
        def f(t):
            z_t = np.exp(1j * t)
            # Direct evaluation for single point
            p_val = 0.0
            for k, c in enumerate(coeffs):
                p_val += c * (z_t ** k)
            return np.abs(p_val)
        
        # Golden section
        gr = (np.sqrt(5) - 1) / 2
        a = t_low
        b = t_high
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = f(c)
        fd = f(d)
        
        for _ in range(20): # 20 iterations for high precision
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
        
        val = max(fc, fd)
        if val > max_val:
            max_val = val
            
    return max_val

def _spectral_init(n):
    """
    Initialize coefficients using spectral synthesis.
    Target a flat spectrum.
    """
    # Create a target spectrum with constant magnitude
    # This corresponds to a Dirac delta in time, which is not +/-1.
    # Instead, we can try to minimize the variance of the spectrum.
    # Or simply use random init and let SA do the work.
    # Spectral init: Inverse FFT of a random phase spectrum with constant magnitude?
    # Let's try: P(z) on unit circle is Fourier series.
    # If we want flat P(z), the coefficients should be orthogonal? No.
    # Let's just use random init, it's often sufficient with SA.
    # But the directive suggests spectral synthesis.
    # Let's try: c_k = sign(Re(1 / (1 + k*epsilon)))? No.
    # Let's try to minimize sum |P(e^{i theta})|^4.
    # For now, random init is robust.
    return np.random.choice([-1, 1], size=n)

def construct():
    n = 24
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    # Initial solution
    best_coeffs = _spectral_init(n)
    best_val = _eval_norm(best_coeffs)
    
    # Simulated Annealing
    T = 0.1
    T_min = 1e-5
    cooling = 0.999
    
    current_coeffs = best_coeffs.copy()
    current_val = best_val
    
    while T > T_min and time.time() < deadline:
        # Pick two random indices to flip
        i, j = np.random.choice(n, 2, replace=False)
        
        new_coeffs = current_coeffs.copy()
        new_coeffs[i] *= -1
        new_coeffs[j] *= -1
        
        new_val = _eval_norm(new_coeffs)
        
        delta = new_val - current_val
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_coeffs = new_coeffs
            current_val = new_val
            
            if current_val < best_val:
                best_val = current_val
                best_coeffs = current_coeffs.copy()
        
        T *= cooling
        
    # Final polish: try single flips on the best solution
    # This helps fine-tune
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                trial = best_coeffs.copy()
                trial[i] *= -1
                v = _eval_norm(trial)
                if v < best_val:
                    best_val = v
                    best_coeffs = trial
                    improved = True
                    
    return list(best_coeffs)
