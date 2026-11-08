import numpy as np
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import time

def _hermite(n, x):
    """Compute physicist's Hermite polynomial H_n(x)"""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    H_nm2 = np.ones_like(x)
    H_nm1 = 2 * x
    for k in range(2, n + 1):
        H_n = 2 * x * H_nm1 - 2 * (k - 1) * H_nm2
        H_nm2 = H_nm1
        H_nm1 = H_n
    return H_nm1

def _compute_c12(c0, c4, c8):
    """Compute c12 from constraint g(0) = 0"""
    # H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=103950
    return -(c0 + 12*c4 + 1680*c8) / 103950

def _compute_bound(c0, c4, c8):
    """Compute ||g||_1 * ||ĝ||_1 for the test function"""
    c12 = _compute_c12(c0, c4, c8)
    
    # Use a fine grid
    N = 100000
    x_max = 15
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]
    
    # Compute H_n(x)
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 11520*x**4 - 13440*x**2 + 1680
    H12 = 4096*x**12 - 73728*x**10 + 451584*x**8 - 1221120*x**6 + 1451520*x**4 - 665280*x**2 + 103950
    
    P = c0*H0 + c4*H4 + c8*H8 + c12*H12
    g = P * np.exp(-x**2/2)
    
    # ||g||_1
    norm_g_1 = np.sum(np.abs(g)) * dx
    
    # Fourier transform using FFT
    # Need to be careful with the FFT convention
    # ĝ(ξ) = ∫ g(x) exp(-2πi x ξ) dx
    # Using FFT: ifft convention is (1/N) Σ x_n exp(2πi nk/N)
    # So fft gives Σ x_n exp(-2πi nk/N)
    
    # For continuous FT via FFT:
    # ĝ(ξ_k) ≈ dx * Σ g(x_j) exp(-2πi x_j ξ_k)
    
    # Using numpy fft: fft(g) = Σ g[j] exp(-2πi j k / N)
    # This corresponds to ξ_k = k / (N * dx) for k in [0, N/2] and negative frequencies for k > N/2
    
    G = np.fft.fft(g) * dx
    
    # Shift to center zero frequency
    G_shifted = np.fft.fftshift(G)
    
    # ||ĝ||_1 = ∫ |ĝ(ξ)| dξ
    # The frequency spacing is dξ = 1 / (N * dx)
    dxi = 1.0 / (N * dx)
    norm_G_1 = np.sum(np.abs(G_shifted)) * dxi
    
    return norm_g_1 * norm_G_1

def construct():
    start_time = time.time()
    deadline = start_time + 180  # 180 seconds max
    
    # Initial guess
    c0_init = -113/100
    c4_init = 1/25
    c8_init = 1/3240
    
    best_result = [c0_init, c4_init, c8_init]
    best_bound = _compute_bound(c0_init, c4_init, c8_init)
    
    # Try optimization
    try:
        def objective(coeffs):
            c0, c4, c8 = coeffs
            try:
                return _compute_bound(c0, c4, c8)
            except:
                return 1e10
        
        # Nelder-Mead is robust
        res = minimize(objective, [c0_init, c4_init, c8_init], method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12,
                                'adaptive': True})
        
        if time.time() < deadline - 30:  # Safety margin
            if res.fun < best_bound:
                best_bound = res.fun
                best_result = list(res.x)
        
        # Try L-BFGS-B
        res2 = minimize(objective, [c0_init, c4_init, c8_init], method='L-BFGS-B',
                        options={'maxiter': 200, 'gtol': 1e-12})
        
        if time.time() < deadline - 30:
            if res2.fun < best_bound:
                best_bound = res2.fun
                best_result = list(res2.x)
        
    except:
        pass
    
    # Check deadline
    if time.time() > deadline - 5:
        return np.array(best_result)
    
    return np.array(best_result)
