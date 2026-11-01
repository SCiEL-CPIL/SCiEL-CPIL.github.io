"""Optimized program: minimize uncertainty bound via Hermite coefficient optimization."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
from math import sqrt, pi, lgamma, log

_START_TIME = time.time()
_DEADLINE = 210.0
_SAFETY = 5.0


def _remaining():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY


def _compute_bound_fast(a0, a4, a8):
    """Compute the uncertainty bound analytically using Hermite polynomial properties.
    
    For g(x) = sum_k a_k H_k(x) e^{-x^2/2}, we need:
    A(g) = ||g||_4 / ||g||_2 and A(F[g]) similarly.
    
    Key: ||g||_2^2 = sum a_k^2 * 2^k k! sqrt(pi)
    For ||g||_4^2 = ∫ |g|^4, we need 4th moments of the Gaussian-weighted Hermite combo.
    
    Using orthogonality: ∫ H_k H_j e^{-x^2} dx = 2^k k! sqrt(pi) δ_kj
    
    For |g|^2 = (sum a_k H_k e^{-x^2/2})^2 = sum_{k,j} a_k a_j H_k H_j e^{-x^2}
    For |g|^4 = |g|^2 * |g|^2, this gets complex. Let me use the fact that:
    
    ∫ H_k H_j H_m H_n e^{-x^2} dx can be computed from known formulas.
    
    Actually, a simpler approach: the L4 norm squared is ∫ g^4 dx (for real g).
    g^4 = (sum a_k H_k e^{-x^2/2})^4 = sum_{k,j,m,n} a_k a_j a_m a_n H_k H_j H_m H_n e^{-2x^2}
    
    This is getting complex. Let me use a semi-analytic approach with precomputed integrals.
    """
    # H_k(0) values for even k
    # H_0(x) = 1, H_4(x) = 16x^4 - 48x^2 + 12, H_8(x) = 256x^8 - 3072x^6 + 10752x^4 - 13440x^2 + 5040
    # H_12(x) = 4096x^12 - 122880x^10 + 1720320x^8 - 10613760x^6 + 31933440x^4 - 40656000x^2 + 10395000
    
    H0_0 = 1.0
    H4_0 = 12.0  # H_4(0) = 12 (wait, let me verify: H_4 = 16x^4-48x^2+12, so H_4(0)=12)
    # Actually physicist's H_4(x) = 16x^4 - 48x^2 + 12, so H_4(0) = 12
    # But some conventions differ. Let me use: H_n(0) = (-1)^{n/2} (n-1)!! for even n
    # H_0(0) = 1, H_2(0) = -2, H_4(0) = 12, H_6(0) = -120, H_8(0) = 1680, H_10(0) = -30240, H_12(0) = 665280
    
    # Let me just use the recurrence to be safe
    def H_val(n, x):
        if n == 0:
            return 1.0
        elif n == 1:
            return 2 * x
        h0, h1 = 1.0, 2 * x
        for k in range(2, n + 1):
            h2 = 2 * x * h1 - 2 * (k - 1) * h0
            h0, h1 = h1, h2
        return h1
    
    H0_0 = H_val(0, 0.0)
    H4_0 = H_val(4, 0.0)
    H8_0 = H_val(8, 0.0)
    H12_0 = H_val(12, 0.0)
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    coeffs = [a0, a4, a8, a12]
    ks = [0, 4, 8, 12]
    
    # ||g||_2^2 = sum a_k^2 * 2^k * k! * sqrt(pi)
    norm2_sq = 0.0
    for ak, k in zip(coeffs, ks):
        norm2_sq += ak ** 2 * (2.0 ** k) * np.exp(lgamma(k + 1)) * sqrt(pi)
    
    if norm2_sq < 1e-30:
        return 1e10
    
    # For ||g||_4^2 = ∫ g^4 dx, use numerical quadrature on a fine grid
    # This is more reliable than trying to compute all 4-index Hermite integrals
    x = np.linspace(-8, 8, 4001)
    dx = x[1] - x[0]
    gauss = np.exp(-x**2 / 2.0)
    
    # Compute H_k(x) for k in {0, 4, 8, 12}
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    # H_8 = 256x^8 - 3072x^6 + 10752x^4 - 13440x^2 + 5040
    x2 = x**2
    H8 = 256*x2**4 - 3072*x2**3 + 10752*x2**2 - 13440*x2 + 5040
    # H_12: use recurrence
    H10 = 2*x*H8 - 14*H4  # H_{10} = 2x H_9 - 18 H_8 ... wait
    # H_{n+1} = 2x H_n - 2n H_{n-1}
    # H_6 = 2x H_5 - 10 H_4
    # Let me compute all needed
    H1 = 2*x
    H2 = 2*x*H1 - 2*H0  # = 4x^2 - 2
    H3 = 2*x*H2 - 4*H1
    H4_check = 2*x*H3 - 6*H2
    H5 = 2*x*H4_check - 8*H3
    H6 = 2*x*H5 - 10*H4_check
    H7 = 2*x*H6 - 12*H5
    H8_check = 2*x*H7 - 14*H6
    H9 = 2*x*H8_check - 16*H7
    H10 = 2*x*H9 - 18*H8_check
    H11 = 2*x*H10 - 20*H9
    H12_check = 2*x*H11 - 22*H10
    
    # Use the recurrence-computed versions for consistency
    g = (a0 * H0 + a4 * H4_check + a8 * H8_check + a12 * H12_check) * gauss
    
    g2 = g ** 2
    g4 = g2 ** 2
    
    norm2_num = np.trapz(g2, x)
    norm4_num = np.trapz(g4, x)
    
    # A(g) = ||g||_4 / ||g||_2
    A_g = (norm4_num ** 0.25) / (norm2_num ** 0.5)
    
    # Fourier transform via FFT
    N = len(x)
    Nfft = 1 << (N - 1).bit_length()
    g_pad = np.zeros(Nfft, dtype=complex)
    g_pad[:N] = g
    G = np.fft.fft(g_pad) * dx
    freqs = np.fft.fftfreq(Nfft, d=dx)
    
    G2 = np.abs(G) ** 2
    G4 = np.abs(G) ** 4
    
    Gnorm2 = np.trapz(G2, freqs)
    Gnorm4 = np.trapz(G4, freqs)
    
    A_G = (Gnorm4 ** 0.25) / (Gnorm2 ** 0.5)
    
    return A_g * A_G


def construct():
    """Return three coefficients (a0, a4, a8) minimizing the uncertainty bound."""
    
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _compute_bound_fast(*best_coeffs)
    
    np.random.seed(42)
    
    def objective(params):
        if _remaining() < 0:
            return best_val
        try:
            val = _compute_bound_fast(params[0], params[1], params[2])
            if val < best_val:
                best_val = val
            return val
        except Exception:
            return 1e10
    
    # Multi-start Nelder-Mead
    n_starts = 6
    for i in range(n_starts):
        if _remaining() < 25.0:
            break
        
        if i == 0:
            x0 = best_coeffs.copy()
        else:
            scale = 0.05 * (1 + i * 0.3)
            x0 = best_coeffs + scale * np.random.randn(3)
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Final polish
    if _remaining() > 15.0:
        try:
            res = minimize(objective, best_coeffs, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
            if res.fun < best_val:
                best_val = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    return np.array(best_coeffs)
# EVOLVE-BLOCK-END
