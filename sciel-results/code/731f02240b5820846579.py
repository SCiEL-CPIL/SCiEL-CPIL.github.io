import numpy as np
from scipy.optimize import minimize
import time

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2 * x
    else:
        # Use recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
        H_nm1 = np.ones_like(x)
        H_n = 2 * x
        for k in range(1, n):
            H_np1 = 2 * x * H_n - 2 * k * H_nm1
            H_nm1 = H_n
            H_n = H_np1
        return H_n

def _compute_bound(c0, c1, c2):
    """Compute the uncertainty product for given coefficients."""
    # Fix c3 so that g(0) = 0
    H0_0 = 1.0
    H4_0 = 3.0
    H8_0 = 105.0
    H12_0 = 10395.0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # Create grid
    x = np.linspace(-10, 10, 10000)
    dx = x[1] - x[0]
    
    H0 = _hermite(0, x)
    H4 = _hermite(4, x)
    H8 = _hermite(8, x)
    H12 = _hermite(12, x)
    
    P = c0 * H0 + c1 * H4 + c2 * H8 + c3 * H12
    g = P * np.exp(-x**2 / 2)
    
    # A(g) = integral of |g|^2
    A_g = np.sum(g**2) * dx
    
    # Fourier transform
    # Use FFT with proper scaling
    N = len(x)
    # For FFT, we need to handle the non-uniform sampling carefully
    # Use the fact that for g(x) = P(x)exp(-x^2/2), the FT is related
    # Let's just compute numerically
    
    # Shift to center for FFT
    g_shifted = np.roll(g, N // 2)
    G = np.fft.fft(g_shifted) * dx
    # Unshift
    G = np.roll(G, -N // 2)
    
    # Frequency axis
    freq = np.fft.fftfreq(N, d=dx)
    # The FT with our convention: ĝ(ξ) = ∫ g(x) e^{-2πixξ} dx
    # numpy fft gives sum g_n exp(-2πi n k/N), so G_k ≈ dx * sum g_n exp(-2πi n k/N)
    # which approximates ∫ g(x) exp(-2πi x ξ_k) dx where ξ_k = k/(N dx)
    
    A_G = np.sum(G**2) * dx  # This is ∫|ĝ|² dξ
    
    # By Plancherel, A_g should equal A_G (up to normalization)
    # The "uncertainty product" might be something else
    
    # Perhaps the bound involves the L1 norm or a different quantity
    # Let's try: the product of the L2 norm and the L2 norm of the derivative
    dg = np.gradient(g, dx)
    A_dg = np.sum(dg**2) * dx
    
    return A_g * A_dg  # This is a common uncertainty measure

def construct():
    start_time = time.time()
    deadline = 180.0  # Safety margin
    
    # Initial best from seed
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_bound = _compute_bound(*best_c)
    
    def objective(c):
        return _compute_bound(c[0], c[1], c[2])
    
    # Use multiple starting points and optimize
    rng = np.random.RandomState(42)
    
    for _ in range(20):
        if time.time() - start_time > deadline - 30:
            break
        
        # Random perturbation around the seed
        scale = 0.1
        c0 = best_c[0] + rng.normal(0, scale)
        c1 = best_c[1] + rng.normal(0, scale)
        c2 = best_c[2] + rng.normal(0, scale)
        
        x0 = np.array([c0, c1, c2])
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                            options={'maxiter': 100, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_bound:
                best_bound = result.fun
                best_c = result.x
        except Exception:
            pass
    
    # Additional refinement with L-BFGS-B
    if time.time() - start_time < deadline - 60:
        try:
            result = minimize(objective, best_c, method='L-BFGS-B',
                            options={'maxiter': 1000, 'ftol': 1e-12})
            if result.fun < best_bound:
                best_bound = result.fun
                best_c = result.x
        except Exception:
            pass
    
    return np.array(best_c)
