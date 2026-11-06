"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
_N = 24
_SQRTN = math.sqrt(_N)
_T0 = time.time()
_LIMIT = 120.0

def _time_left():
    return _LIMIT - (time.time() - _T0)

def _rs_poly():
    """Rudin-Shapiro coefficients truncated to _N."""
    p, q = [1], [1]
    while len(p) < _N:
        p, q = p + q, p + [-c for c in q]
    return p[:_N]

def _eval(coeffs):
    """Dense FFT sampling to estimate max |P(z)| on unit circle."""
    n = 4096
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.exp(1j * angles)
    # P(z) = sum c_k z^k
    # Use polynomial evaluation via FFT-friendly method
    # Pad coefficients to length n
    c = np.zeros(n, dtype=complex)
    c[:len(coeffs)] = coeffs
    # Evaluate P(z) at all n points
    # P(z_j) = sum_k c_k z_j^k = sum_k c_k exp(i * 2*pi*j*k/n)
    # This is the DFT of c (with appropriate sign convention)
    # DFT: X_j = sum_k c_k exp(-2*pi*i*j*k/n)
    # We want exp(+2*pi*i*j*k/n), so use inverse DFT or conjugate
    # Actually, let's just use numpy's polynomial evaluation or direct
    # For efficiency, use the fact that this is a DFT
    # P(z_j) = sum_k c_k exp(2*pi*i*j*k/n) = IDFT(c) * n? No...
    # Let's use np.polyval style or just FFT
    # c_k at index k, we want sum c_k * exp(2*pi*i*j*k/n)
    # This is the DFT with + sign, which is n * IDFT(c)
    vals = np.fft.ifft(c) * n
    return np.max(np.abs(vals))

def _score_final(coeffs):
    """Compute the final score with golden-section refinement."""
    # Dense sampling
    n = 8192
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = np.exp(1j * angles)
    c = np.zeros(n, dtype=complex)
    c[:len(coeffs)] = coeffs
    vals = np.fft.ifft(c) * n
    mag = np.abs(vals)
    
    # Find top peaks and refine
    peak_indices = np.argsort(mag)[-10:]
    max_val = mag[peak_indices[0]]
    
    # Golden section refinement around top peaks
    golden = (1 + math.sqrt(5)) / 2
    for idx in peak_indices[:5]:
        # Refine around angles[idx]
        a = angles[(idx - 1) % n]
        b = angles[(idx + 1) % n]
        # Ensure a < b
        if a > b:
            a, b = b, a
        # Golden section search
        for _ in range(30):
            m1 = b - (b - a) / golden
            m2 = a + (b - a) / golden
            f1 = np.abs(np.sum(coeffs * np.exp(1j * m1 * np.arange(_N))))
            f2 = np.abs(np.sum(coeffs * np.exp(1j * m2 * np.arange(_N))))
            if f1 < f2:
                b = m2
            else:
                a = m1
        f_mid = np.abs(np.sum(coeffs * np.exp(1j * (a + b) / 2 * np.arange(_N))))
        max_val = max(max_val, f_mid)
    
    return max_val / _SQRTN

def _flip_improve(coeffs, max_iter=5000):
    """Local search: flip coefficients to reduce peak."""
    coeffs = list(coeffs)
    best_score = _score_final(coeffs)
    n_flips = 0
    
    for _ in range(max_iter):
        if _time_left() < 5.0:
            break
        
        improved = False
        # Try flipping each coefficient
        for k in range(_N):
            old = coeffs[k]
            coeffs[k] = -old
            score = _score_final(coeffs)
            if score < best_score - 1e-12:
                best_score = score
                improved = True
                n_flips += 1
            else:
                coeffs[k] = old
        
        if not improved:
            break
    
    return coeffs, best_score

def construct():
    # Start with Rudin-Shapiro
    coeffs = _rs_poly()
    
    # Local search to improve
    coeffs, score = _flip_improve(coeffs)
    
    return coeffs
# EVOLVE-BLOCK-END
