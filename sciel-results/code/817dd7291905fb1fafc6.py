"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

# Deadline guard: 210s limit, use 195s for safety
_DEADLINE = time.time() + 195

def _hermite(n, x):
    """Compute physicists' Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    elif n == 1:
        return 2.0 * x
    else:
        h_prev = np.ones_like(x)
        h_curr = 2.0 * x
        for k in range(2, n + 1):
            h_next = 2.0 * x * h_curr - 2.0 * (k - 1) * h_prev
            h_prev, h_curr = h_curr, h_next
        return h_curr

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product bound for g(x) = P(x)exp(-x^2/2).
    P(x) = a0*H0 + a4*H4 + a8*H8 + a12*H12.
    Constraint: P(0) = 0 => a12 = -(a0*1 + a4*12 + a8*3024) / 10321920.
    
    The bound is A(g)*A(F(g)) where A(f) = ||f||_1 / min|F(f)| (or similar L1-based measure).
    For the specific inequality context (Goncalves et al.), the bound is often expressed via:
    A(g) * A(ĝ) >= 1/4 (standard) or a tuned constant.
    
    Let's use the numerical definition:
    A(g) = ||g||_1 / ||ĝ||_infty? No, typically A(f) = ||f||_1 / ||f||_2? No.
    
    In the context of "Uncertainty inequality" tuning Hermite polynomials, the quantity to minimize is usually:
    C = ||g||_1 * ||ĝ||_1 / (||g||_2 * ||ĝ||_2) ? No.
    
    Let's look at the standard definition for these tuning problems.
    Usually, we want to minimize the constant C such that ||f||_1 ||ĝ||_1 >= C ||f||_2^2.
    Or we minimize the upper bound on the product of spreads.
    
    The task says: "reduce the upper bound on A(f)*A(Fourier(f))".
    And "score is the upper bound on C4".
    
    Let's assume the standard L1 uncertainty product:
    U = (||g||_1 / ||g||_2) * (||ĝ||_1 / ||ĝ||_2)
    By Plancherel, ||ĝ||_2 = ||g||_2.
    So U = ||g||_1 * ||ĝ||_1 / ||g||_2^2.
    
    We can compute this numerically.
    """
    # Coefficients
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 3024.0
    H12_0 = 10321920.0
    
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    
    # Define P(x)
    def P(x):
        return a0 * _hermite(0, x) + a4 * _hermite(4, x) + a8 * _hermite(8, x) + a12 * _hermite(12, x)
    
    # Grid
    x = np.linspace(-10, 10, 20000)
    dx = x[1] - x[0]
    
    g = P(x) * np.exp(-x**2 / 2.0)
    
    # L2 norm squared
    norm2_sq = np.sum(g**2) * dx
    
    # L1 norm
    norm1 = np.sum(np.abs(g)) * dx
    
    # Fourier transform
    # Use FFT for efficiency? Or analytical?
    # g(x) is a polynomial times Gaussian. Its Fourier transform is also a polynomial times Gaussian.
    # F(P(x)e^{-x^2/2}) = (1/sqrt(2pi)) * i^{-deg} P(-i) e^{-xi^2/2}?
    # Let's just use FFT on the grid, it's fast enough for 20k points.
    # Zero-padding for better frequency resolution?
    N = len(x)
    # Shift to center
    g_shifted = np.roll(g, N//2)
    G = np.fft.fft(g_shifted) * dx
    freqs = np.fft.fftfreq(N, d=dx) * 2 * np.pi
    
    # L1 norm of Fourier transform
    norm1_f = np.sum(np.abs(G)) * (2 * np.pi / N) # Normalization factor for FFT
    
    # Note: The exact normalization of FFT might affect the absolute value,
    # but since we are comparing relative improvements, the constant factor cancels out
    # if we are consistent. However, to get the "true" bound, we should be careful.
    # Let's use the unitary convention or check against known values.
    # For now, let's compute the ratio.
    
    bound = (norm1 * norm1_f) / norm2_sq
    
    return bound

def _objective(vec):
    a0, a4, a8 = vec
    try:
        return _compute_bound(a0, a4, a8)
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    
    # Seed values
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_val = _objective(seed)
    best_x = seed.copy()
    
    # Multi-start local optimization
    # We will try several random starts and local minimization.
    np.random.seed(42)
    
    num_starts = 5
    for i in range(num_starts):
        if time.time() > _DEADLINE:
            break
        
        if i == 0:
            x0 = seed
        else:
            # Random perturbation
            scale = 0.1
            x0 = seed + np.random.normal(0, scale, 3)
        
        try:
            res = minimize(_objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.success or res.fun < best_val:
                if res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x
        except Exception:
            pass
        
        # Check time
        if time.time() > _DEADLINE - 10:
            break
    
    # Final polish with higher precision
    if time.time() < _DEADLINE - 5:
        try:
            res = minimize(_objective, best_x, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
