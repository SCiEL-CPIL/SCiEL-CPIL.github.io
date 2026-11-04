"""Second autocorrelation inequality construction.

Strategy: Use DFT parameterization to enforce non-negativity implicitly via
h = |IDFT(g)|^2, and optimize the complex DFT coefficients directly.
This is a structural change from the time-domain complex parameterization.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 50

def _compute_ratio(h):
    """Compute ||h||_2^2 / (||h||_1 * ||h*h||_inf) for nonnegative h."""
    h = np.asarray(h, dtype=np.float64)
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = np.correlate(h_norm, h_norm, mode='full')
    conv_max = np.max(conv)
    if conv_max <= 0:
        return 0.0
    return l2_sq / conv_max

def _dft_parameterize(g_complex):
    """Convert DFT coefficients to time-domain nonnegative h.
    g_complex is the DFT of h, so h = IDFT(g_complex).
    We use h = |IDFT(g_complex)|^2 to ensure non-negativity.
    """
    h_complex = np.fft.ifft(g_complex)
    h = np.abs(h_complex) ** 2
    return h

def _objective_dft(g_flat):
    """Objective: negative ratio for DFT parameterization."""
    g_complex = g_flat[0::2] + 1j * g_flat[1::2]
    h = _dft_parameterize(g_complex)
    return -_compute_ratio(h)

def _smooth_max(x, beta=1000.0):
    """Smooth approximation of max using log-sum-exp."""
    x = np.asarray(x, dtype=np.float64)
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(beta * (x - x_max)))) / beta

def _objective_dft_smooth(g_flat, beta=1000.0):
    """Smoothed objective for better gradient behavior."""
    g_complex = g_flat[0::2] + 1j * g_flat[1::2]
    h_complex = np.fft.ifft(g_complex)
    h = np.abs(h_complex) ** 2
    l1 = np.sum(h)
    if l1 <= 0:
        return 0.0
    h_norm = h / l1
    l2_sq = np.sum(h_norm ** 2)
    conv = np.correlate(h_norm, h_norm, mode='full')
    conv_smooth = _smooth_max(conv, beta)
    if conv_smooth <= 0:
        return 0.0
    return -l2_sq / conv_smooth

def _find_good_dft_init():
    """Generate a good initial DFT coefficient vector."""
    # Start with random phases and magnitudes
    np.random.seed(42)
    
    # Try a few different initializations
    inits = []
    
    # 1. Random complex
    g = (np.random.randn(N) + 1j * np.random.randn(N)).astype(np.complex128)
    inits.append(g)
    
    # 2. Low-pass filter (smooth in frequency)
    freqs = np.fft.fftfreq(N)
    mask = np.abs(freqs) < 0.3
    g = (np.random.randn(N) + 1j * np.random.randn(N)) * mask
    g = g.astype(np.complex128)
    inits.append(g)
    
    # 3. Band-pass
    mask = (np.abs(freqs) > 0.1) & (np.abs(freqs) < 0.4)
    g = (np.random.randn(N) + 1j * np.random.randn(N)) * mask
    g = g.astype(np.complex128)
    inits.append(g)
    
    # 4. All ones (uniform in time)
    g = np.ones(N, dtype=np.complex128)
    inits.append(g)
    
    # 5. Sparse DFT
    support = np.random.choice(N, size=10, replace=False)
    g = np.zeros(N, dtype=np.complex128)
    g[support] = np.random.randn(10) + 1j * np.random.randn(10)
    inits.append(g)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_score = 0.0
    best_h = np.ones(N) / N
    
    # Initial best
    best_score = _compute_ratio(best_h)
    
    # Try DFT parameterization with multiple initializations
    dft_inits = _find_good_dft_init()
    
    for trial, g_init in enumerate(dft_inits):
        if time.time() > deadline:
            break
        
        # Flatten to real vector
        g_flat_init = np.zeros(2 * N)
        g_flat_init[0::2] = g_init.real
        g_flat_init[1::2] = g_init.imag
        
        try:
            # Use smooth objective for better convergence
            res = minimize(
                _objective_dft_smooth,
                g_flat_init,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 50000, 'ftol': 1e-12}
            )
            
            g_opt_flat = res.x
            g_opt = g_opt_flat[0::2] + 1j * g_opt_flat[1::2]
            h_opt = _dft_parameterize(g_opt)
            
            # Normalize
            l1 = np.sum(h_opt)
            if l1 > 0:
                h_opt = h_opt / l1
            
            s = _compute_ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Polish pass: optimize in time domain using the best DFT result
    if time.time() < deadline - 30:
        try:
            # Convert best_h to DFT coefficients
            g_best = np.fft.fft(best_h)
            g_flat_init = np.zeros(2 * N)
            g_flat_init[0::2] = g_best.real
            g_flat_init[1::2] = g_best.imag
            
            res = minimize(
                _objective_dft_smooth,
                g_flat_init,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'maxfun': 100000, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            
            g_opt_flat = res.x
            g_opt = g_opt_flat[0::2] + 1j * g_opt_flat[1::2]
            h_opt = _dft_parameterize(g_opt)
            
            l1 = np.sum(h_opt)
            if l1 > 0:
                h_opt = h_opt / l1
            
            s = _compute_ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    # Final polish: direct time-domain optimization
    if time.time() < deadline - 10:
        try:
            def _objective_time(h_flat):
                h = np.maximum(h_flat, 0)
                return -_compute_ratio(h)
            
            def _jac_time(h_flat):
                eps = 1e-8
                grad = np.zeros(N)
                for i in range(N):
                    h_plus = h_flat.copy()
                    h_plus[i] += eps
                    h_minus = h_flat.copy()
                    h_minus[i] -= eps
                    grad[i] = (_objective_time(h_plus) - _objective_time(h_minus)) / (2 * eps)
                return grad
            
            res = minimize(
                _objective_time,
                best_h.copy(),
                method='L-BFGS-B',
                jac=_jac_time,
                options={'maxiter': 500, 'maxfun': 20000, 'ftol': 1e-14}
            )
            
            h_opt = np.maximum(res.x, 0)
            l1 = np.sum(h_opt)
            if l1 > 0:
                h_opt = h_opt / l1
            
            s = _compute_ratio(h_opt)
            if s > best_score:
                best_score = s
                best_h = h_opt.copy()
                
        except Exception:
            pass
    
    best_h = np.maximum(best_h, 0)
    return list(best_h)
