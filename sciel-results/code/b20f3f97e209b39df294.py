import numpy as np
import time

def _eval_sup_norm(coeffs, N=4096):
    """Evaluate sup norm of polynomial on unit circle with golden-section refinement."""
    n = len(coeffs)
    coeffs = np.array(coeffs, dtype=np.float64)
    
    # Dense sampling
    theta = 2 * np.pi * np.arange(N) / N
    z = np.exp(1j * theta)
    
    # Evaluate polynomial: P(z) = sum c_k z^k
    # Use FFT-based approach: pad coeffs to length N, compute IFFT
    # Actually for small n, direct matrix multiply is fine
    # P(z_j) = sum_k c_k * exp(i*k*theta_j)
    # This is like DFT of c (padded)
    c_padded = np.zeros(N, dtype=np.complex128)
    c_padded[:n] = coeffs
    P = np.fft.ifft(c_padded) * N  # ifft gives (1/N) sum, so multiply by N
    # Wait, let me be more careful.
    # DFT: X_k = sum_j x_j exp(-2pi i jk/N)
    # We want P(theta_j) = sum_k c_k exp(i k theta_j) = sum_k c_k exp(2pi i k j / N)
    # This is the inverse DFT of c (padded to N): 
    # IFFT(c)_j = (1/N) sum_k c_k exp(2pi i k j / N)
    # So P(theta_j) = N * IFFT(c_padded)_j
    mags = np.abs(P)
    
    # Find peaks and refine with golden section
    # Get top 20 peak indices
    top_k = min(20, N)
    peak_idx = np.argsort(mags)[-top_k:][::-1]
    
    best_mag = 0.0
    for idx in peak_idx:
        # Refine around theta[idx] using golden section
        t0 = theta[idx]
        # Search in small neighborhood
        delta = 2 * np.pi / N
        a, b = t0 - delta, t0 + delta
        
        def f(t):
            zt = np.exp(1j * t)
            # Evaluate polynomial at single point
            # P(zt) = sum c_k zt^k
            powers = zt ** np.arange(n)
            val = np.dot(coeffs, powers)
            return np.abs(val)
        
        # Golden section search
        gr = (np.sqrt(5) - 1) / 2  # ~0.618
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = f(c_pt)
        fd = f(d_pt)
        
        for _ in range(50):
            if fc < fd:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = f(c_pt)
            else:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = f(d_pt)
        
        refined_val = max(fc, fd)
        if refined_val > best_mag:
            best_mag = refined_val
    
    return best_mag

def _try_flip(coeffs, i, N=4096):
    """Try flipping coefficient i and return new norm if it improves."""
    new_coeffs = list(coeffs)
    new_coeffs[i] = -new_coeffs[i]
    return _eval_sup_norm(new_coeffs, N)

def construct():
    deadline = time.time() + 140  # 140s to leave 10s margin
    
    # Start with Golay-Rudin-Shapiro
    def grs_coeffs(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    
    best_coeffs = grs_coeffs(24)
    best_norm = _eval_sup_norm(best_coeffs)
    
    # Try different variants: shift, sign flip
    variants = [best_coeffs]
    # Shift by 1
    shifted = best_coeffs[1:] + [best_coeffs[0]]
    variants.append(shifted)
    # Sign flip all
    variants.append([-c for c in best_coeffs])
    # Sign flip all + shift
    variants.append([-c for c in shifted])
    
    # Evaluate all variants
    for v in variants:
        nm = _eval_sup_norm(v)
        if nm < best_norm:
            best_norm = nm
            best_coeffs = list(v)
    
    # Local search: single flips
    n = 24
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            new_norm = _try_flip(best_coeffs, i)
            if new_norm < best_norm - 1e-12:
                best_coeffs[i] = -best_coeffs[i]
                best_norm = new_norm
                improved = True
    
    # Local search: pair flips
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                if time.time() > deadline - 5:
                    break
                # Try flipping both i and j
                old_i, old_j = best_coeffs[i], best_coeffs[j]
                best_coeffs[i] = -old_i
                best_coeffs[j] = -old_j
                new_norm = _eval_sup_norm(best_coeffs)
                if new_norm < best_norm - 1e-12:
                    best_norm = new_norm
                    improved = True
                else:
                    best_coeffs[i] = old_i
                    best_coeffs[j] = old_j
            if time.time() > deadline - 5:
                break
    
    # Final polish: more iterations of single flips with tighter tolerance
    for _ in range(5):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 1:
                break
            new_norm = _try_flip(best_coeffs, i)
            if new_norm < best_norm - 1e-15:
                best_coeffs[i] = -best_coeffs[i]
                best_norm = new_norm
                improved = True
        if not improved:
            break
    
    # Ensure all coefficients are ±1
    best_coeffs = [1 if c > 0 else -1 for c in best_coeffs]
    
    return best_coeffs
