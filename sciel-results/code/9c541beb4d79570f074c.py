"""DCT-basis spectral shaping: optimize coefficients to minimize max autoconvolution peak."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.fft import dct, idct

N = 400
DEADLINE = 145.0
K = 20  # number of DCT basis vectors to use


def _autocorr_fft(h):
    """Fast full autoconvolution via FFT."""
    n = len(h)
    nfft = 2 ** (2 * n - 1).bit_length()
    H = np.fft.rfft(h, n=nfft)
    ac = np.fft.irfft(H * np.conj(H), n=nfft)[:2 * n - 1]
    return ac


def _value(h):
    S = float(np.sum(h))
    if S == 0.0:
        return np.inf
    c = _autocorr_fft(h)
    return abs(2.0 * N * float(np.max(c)) / (S * S))


def _h_from_dct_coeffs(coeffs):
    """Reconstruct h from DCT-II coefficients (first K)."""
    # DCT-II is orthonormal with type=2, norm='ortho'
    # We need to handle the DC term carefully
    h = np.zeros(N)
    for k in range(min(K, N)):
        if k == 0:
            # DC component: idct gives back the coefficient scaled
            h += coeffs[0] * np.sqrt(1.0 / N)
        else:
            # For k > 0, the DCT basis is sqrt(2/N) * cos(pi * k * (i + 0.5) / N)
            i = np.arange(N)
            basis = np.sqrt(2.0 / N) * np.cos(np.pi * k * (i + 0.5) / N)
            h += coeffs[k] * basis
    return h


def _obj_dct(coeffs):
    """Objective: value computed from DCT coefficients."""
    h = _h_from_dct_coeffs(coeffs)
    return _value(h)


def construct():
    t0 = time.time()
    
    # Start with a reasonable initial point
    # Use ones as baseline, get its DCT coefficients
    h_init = np.ones(N)
    dct_init = dct(h_init, type=2, norm='ortho')
    
    # Truncate to K coefficients
    coeffs_init = np.zeros(K)
    coeffs_init[:min(K, N)] = dct_init[:min(K, N)]
    
    best_coeffs = coeffs_init.copy()
    best_v = _obj_dct(best_coeffs)
    
    # Try multiple random restarts with L-BFGS-B
    np.random.seed(42)
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Random perturbation of coefficients
        if r == 0:
            coeffs_start = best_coeffs.copy()
        else:
            # Scale random coefficients
            scale = np.random.uniform(0.5, 2.0)
            coeffs_start = np.random.randn(K) * scale
            # Ensure DC component is reasonable (positive sum)
            coeffs_start[0] = np.abs(coeffs_start[0]) + 1.0
        
        try:
            res = minimize(
                _obj_dct,
                coeffs_start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50},
            )
            if res.x is not None:
                v = _obj_dct(res.x)
                if v < best_v:
                    best_v = v
                    best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final safety check
    if time.time() - t0 > DEADLINE - 2:
        return np.ones(N)
    
    return _h_from_dct_coeffs(best_coeffs)
