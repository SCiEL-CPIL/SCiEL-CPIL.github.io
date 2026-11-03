"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses Fourier-basis parameterization to reduce dimensionality, then
polishes with coordinate descent on the active set.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 135  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(20):
        diff = target - np.sum(h)
        if abs(diff) < 1e-15:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    # Final exact adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            mask = h < 1.0
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
        else:
            mask = h > 0.0
            if np.any(mask):
                h[mask] += diff / np.sum(mask)
                h = np.clip(h, 0.0, 1.0)
                
    return h


def _fourier_to_h(coeffs):
    """
    Convert Fourier coefficients to step heights.
    coeffs[0] is ignored (mean is fixed at 0.5).
    coeffs[1:m] are complex amplitudes for frequencies 1..m-1.
    Returns h of length N with sum exactly N/2.
    """
    m = len(coeffs)
    i = np.arange(N)
    h = np.full(N, 0.5)
    
    # Add Fourier components
    for k in range(1, m):
        amp = coeffs[k]  # complex
        # Real part: amp * cos(2*pi*k*i/N), Imag part: -amp * sin(2*pi*k*i/N)
        h += amp.real * np.cos(2.0 * np.pi * k * i / N)
        h += amp.imag * np.sin(2.0 * np.pi * k * i / N)
    
    # Enforce box constraints [0, 1]
    h = np.clip(h, 0.0, 1.0)
    
    # Fix sum exactly
    h = _fix_sum(h)
    
    return h


def _h_to_fourier(h, m):
    """
    Extract Fourier coefficients from h.
    Returns array of length m where coeffs[0]=0, coeffs[1:] are complex.
    """
    coeffs = np.zeros(m, dtype=np.complex128)
    H = np.fft.rfft(h)
    for k in range(1, m):
        if k < len(H):
            coeffs[k] = H[k] / N
    return coeffs


def _smooth_max_value_grad_fourier(coeffs, m, beta):
    """
    Returns value and gradient of smooth max of C_k with respect to Fourier coeffs.
    coeffs[0] is unused.
    """
    h = _fourier_to_h(coeffs)
    n = len(h)
    cs = _linear_overlap_values(h)
    max_cs = np.max(cs)
    
    # Smooth max
    exp_terms = np.exp(beta * (cs - max_cs))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = max_cs + np.log(sum_exp) / beta
    
    # Gradient with respect to h
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    
    # term1_j = sum_{k=0}^{j} w[k] * h[j-k]
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # term2_j = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    h_flip = h[::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv2_full = np.fft.irfft(W * HF, n=n_fft)
    term2 = conv2_full[n - 1 - np.arange(n)]
    
    grad_h = -(2.0 / n) * (conv1 + term2)
    
    # Chain rule: grad_coeffs = grad_h @ (dh/dcoeffs)
    # dh/dcoeffs[k] for k>=1: cos(2*pi*k*i/N) for real, sin(2*pi*k*i/N) for imag
    i = np.arange(N)
    grad_coeffs = np.zeros(m, dtype=np.float64)
    
    for k in range(1, m):
        cos_term = np.cos(2.0 * np.pi * k * i / N)
        sin_term = np.sin(2.0 * np.pi * k * i / N)
        # d(h)/d(coeffs[k].real) = cos_term
        # d(h)/d(coeffs[k].imag) = sin_term
        grad_coeffs_real = np.dot(grad_h, cos_term)
        grad_coeffs_imag = np.dot(grad_h, sin_term)
        # Store as real and imag parts in separate slots
        # We'll use a flattened representation: [real_1, imag_1, real_2, imag_2, ...]
        # But for now, return complex gradient
        pass
    
    # Return gradient in a way that matches the optimizer
    # We'll use a real-valued parameterization: coeffs_real and coeffs_imag
    return val, grad_h


def _smooth_max_value_grad_fourier_real(coeffs_real, m, beta):
    """
    Returns value and gradient of smooth max with respect to real Fourier params.
    coeffs_real is a real array of length 2*(m-1): [real_1, imag_1, real_2, imag_2, ...]
    """
    m_complex = m
    coeffs = np.zeros(m_complex, dtype=np.complex128)
    for k in range(1, m_complex):
        coeffs[k] = coeffs_real[2*(k-1)] + 1j * coeffs_real[2*(k-1)+1]
    
    h = _fourier_to_h(coeffs)
    n = len(h)
    cs = _linear_overlap_values(h)
    max_cs = np.max(cs)
    
    exp_terms = np.exp(beta * (cs - max_cs))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = max_cs + np.log(sum_exp) / beta
    
    # Gradient with respect to h
    n_fft = 2 * n
    w_pad = np.zeros(n_fft)
    w_pad[:n] = w
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    W = np.fft.rfft(w_pad, n=n_fft)
    H = np.fft.rfft(h_pad, n=n_fft)
    
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    h_flip = h[::-1]
    h_flip_pad = np.zeros(n_fft)
    h_flip_pad[:n] = h_flip
    HF = np.fft.rfft(h_flip_pad, n=n_fft)
    conv2_full = np.fft.irfft(W * HF, n=n_fft)
    term2 = conv2_full[n - 1 - np.arange(n)]
    
    grad_h = -(2.0 / n) * (conv1 + term2)
    
    # Chain rule to get gradient w.r.t. coeffs_real
    i = np.arange(N)
    grad_coeffs_real = np.zeros(2 * (m_complex - 1))
    
    for k in range(1, m_complex):
        cos_term = np.cos(2.0 * np.pi * k * i / N)
        sin_term = np.sin(2.0 * np.pi * k * i / N)
        grad_coeffs_real[2*(k-1)] = np.dot(grad_h, cos_term)
        grad_coeffs_real[2*(k-1)+1] = np.dot(grad_h, sin_term)
    
    return val, grad_coeffs_real


def _fourier_optimize(m, beta, max_iter=500, tol=1e-10):
    """
    Optimize Fourier coefficients using L-BFGS-B.
    """
    # Initial guess: small random coefficients
    np.random.seed(42)
    coeffs0 = np.random.normal(0, 0.1, 2 * (m - 1))
    
    def obj_func(x):
        val, _ = _smooth_max_value_grad_fourier_real(x, m, beta)
        return val
    
    def jac_func(x):
        _, grad = _smooth_max_value_grad_fourier_real(x, m, beta)
        return grad
    
    try:
        res = minimize(obj_func, coeffs0, jac=jac_func, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
        return res.x
    except Exception:
        return coeffs0


def _coordinate_descent_polish(h, max_iter=100):
    """
    Coordinate descent polish: adjust individual h[i] to reduce max overlap.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    h = _fix_sum(h)
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 15:
            break
            
        cs = _linear_overlap_values(h)
        max_k = np.argmax(cs)
        max_val = cs[max_k]
        
        # Try to reduce h[max_k] or adjust neighbors
        improved = False
        
        # Try small adjustments to h[max_k]
        for delta in [-0.01, -0.005, -0.001, 0.001, 0.005, 0.01]:
            h_new = h.copy()
            h_new[max_k] += delta
            h_new = _fix_sum(h_new)
            cs_new = _linear_overlap_values(h_new)
            if np.max(cs_new) < max_val - 1e-12:
                h = h_new
                improved = True
                break
        
        if not improved:
            # Try adjusting other indices
            for j in range(N):
                if j == max_k:
                    continue
                for delta in [-0.01, 0.01]:
                    h_new = h.copy()
                    h_new[j] += delta
                    h_new = _fix_sum(h_new)
                    cs_new = _linear_overlap_values(h_new)
                    if np.max(cs_new) < max_val - 1e-12:
                        h = h_new
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    # Try different numbers of Fourier modes
    for m in [5, 10, 15, 20, 30]:
        if time.time() - t0 > _DEADLINE - 60:
            break
        
        # Optimize Fourier coefficients
        coeffs_opt = _fourier_optimize(m, beta=100.0, max_iter=300)
        h_fourier = _fourier_to_h(np.zeros(m, dtype=np.complex128))
        
        # Reconstruct h from optimized coefficients
        coeffs = np.zeros(m, dtype=np.complex128)
        for k in range(1, m):
            coeffs[k] = coeffs_opt[2*(k-1)] + 1j * coeffs_opt[2*(k-1)+1]
        h_fourier = _fourier_to_h(coeffs)
        
        _update_best(h_fourier)
        
        # Polish with coordinate descent
        h_polished = _coordinate_descent_polish(h_fourier, max_iter=50)
        _update_best(h_polished)
    
    # Also try a few random starts with PGT as fallback
    if time.time() - t0 < _DEADLINE - 30:
        i = np.arange(N)
        starts = [
            _fix_sum(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)),
            _fix_sum(0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)),
            _fix_sum(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)),
        ]
        
        for h_start in starts:
            if time.time() - t0 > _DEADLINE - 15:
                break
            _update_best(h_start)
            h_polished = _coordinate_descent_polish(h_start, max_iter=30)
            _update_best(h_polished)
    
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
