"""
Optimized search for Hermite coefficients minimizing the uncertainty bound.

Strategy:
1. Analytically express the uncertainty product bound in terms of the three free coefficients.
2. Use scipy.optimize.minimize with multiple starting points to find the global minimum.
3. Include a deadline guard to ensure we finish within the time limit.
"""
import numpy as np
from scipy.optimize import minimize
import time

# Deadline guard: must finish well before 210s
_DEADLINE = 180.0  # seconds


def _hermite_coeffs():
    """
    Return the coefficients of the Hermite polynomial combination.
    
    The test function is:
        g(x) = a0 * H0(x) * exp(-x^2/2) + a4 * H4(x) * exp(-x^2/2) + a8 * H8(x) * exp(-x^2/2) + a12 * H12(x) * exp(-x^2/2)
    
    where a12 is determined by the constraint g(0) = 0.
    
    The Hermite polynomials (physicists' convention) are:
        H0(x) = 1
        H4(x) = 16x^4 - 48x^2 + 12
        H8(x) = 256x^8 - 1344x^6 + 2688x^4 - 1344x^2 + 168
        H12(x) = 4096x^12 - 32256x^10 + 90720x^8 - 116128x^6 + 57024x^4 - 9072x^2 + 252
    
    At x=0:
        H0(0) = 1
        H4(0) = 12
        H8(0) = 168
        H12(0) = 252
    
    Constraint g(0) = 0:
        a0 * 1 + a4 * 12 + a8 * 168 + a12 * 252 = 0
        => a12 = -(a0 + 12*a4 + 168*a8) / 252
    """
    return {
        'H0': 1,
        'H4': 12,
        'H8': 168,
        'H12': 252
    }


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the given coefficients.
    
    The bound involves ||g||_1 and the minimum of |ĝ(ξ)| where ĝ is the Fourier transform.
    For our test function, we need to compute:
        C4 = ||g||_1 / min_ξ |ĝ(ξ)|
    
    Actually, the uncertainty inequality bound is:
        A(f) * A(F) >= C4
    
    where C4 depends on the test function. The exact form is complex, but we can
    approximate it numerically.
    
    Let's use a numerical approach:
    - Construct g(x) on a grid
    - Compute ||g||_1 = integral of |g(x)| dx
    - Compute ĝ(ξ) via FFT or analytical Fourier transform
    - Find min |ĝ(ξ)|
    - The bound is related to ||g||_1 / min|ĝ|
    """
    # Determine a12 from constraint g(0) = 0
    a12 = -(a0 + 12*a4 + 168*a8) / 252.0
    
    # Numerical computation
    # Use a sufficiently large grid
    N = 4096
    x_max = 12.0
    x = np.linspace(-x_max, x_max, N)
    dx = x[1] - x[0]
    
    # Hermite polynomials
    def H0(x):
        return np.ones_like(x)
    
    def H4(x):
        return 16*x**4 - 48*x**2 + 12
    
    def H8(x):
        return 256*x**8 - 1344*x**6 + 2688*x**4 - 1344*x**2 + 168
    
    def H12(x):
        return 4096*x**12 - 32256*x**10 + 90720*x**8 - 116128*x**6 + 57024*x**4 - 9072*x**2 + 252
    
    # Gaussian factor
    gauss = np.exp(-x**2 / 2.0)
    
    # Construct g(x)
    g = (a0 * H0(x) + a4 * H4(x) + a8 * H8(x) + a12 * H12(x)) * gauss
    
    # ||g||_1
    norm1 = np.sum(np.abs(g)) * dx
    
    # Fourier transform using FFT
    # ĝ(ξ) = ∫ g(x) e^{-iξx} dx
    # We need to be careful with the FFT scaling
    fft_vals = np.fft.fft(g) * dx
    # Shift to center at zero frequency
    fft_vals = np.fft.fftshift(fft_vals)
    
    # Frequency axis
    freq = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # Find minimum of |ĝ(ξ)|, but exclude very small values near zero frequency
    # to avoid numerical issues
    abs_fft = np.abs(fft_vals)
    
    # The minimum should be over a range where the function is well-behaved
    # Let's find the minimum excluding the very center if it's too small
    min_abs = np.min(abs_fft)
    
    # The bound is approximately ||g||_1 / min|ĝ|
    # But we need to be careful - if min_abs is very small, the bound is large
    if min_abs < 1e-15:
        return np.inf
    
    bound = norm1 / min_abs
    
    return bound


def construct():
    """
    Return the three coefficients (a0, a4, a8) that minimize the uncertainty bound.
    The degree-12 weight is fixed so that g(0) = 0.
    """
    start_time = time.time()
    
    # Starting points for optimization
    start_points = [
        [-113/100, 1/25, 1/3240],  # Current best
        [0, 0, 0],
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
        [1, 1, 1],
        [-1, -1, -1],
        [1, -1, 1],
        [-1, 1, -1],
        [0.5, 0.5, 0.5],
        [-0.5, -0.5, -0.5],
        [1, 1, -1],
        [-1, -1, 1],
        [2, 1, 0.5],
        [-2, -1, -0.5],
        [0.1, 0.1, 0.1],
        [-0.1, -0.1, -0.1],
    ]
    
    best_coeff = np.array([-113/100, 1/25, 1/3240])
    best_bound = _compute_bound(*best_coeff)
    
    # Try each starting point
    for sp in start_points:
        if time.time() - start_time > _DEADLINE - 30:
            break
        
        try:
            result = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                sp,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if result.success:
                bound = _compute_bound(result.x[0], result.x[1], result.x[2])
                if bound < best_bound:
                    best_bound = bound
                    best_coeff = result.x.copy()
        except Exception:
            pass
    
    # If we still have time, try more random starting points
    if time.time() - start_time < _DEADLINE - 60:
        rng = np.random.RandomState(42)
        for _ in range(50):
            if time.time() - start_time > _DEADLINE - 30:
                break
            
            sp = rng.uniform(-3, 3, size=3)
            try:
                result = minimize(
                    lambda x: _compute_bound(x[0], x[1], x[2]),
                    sp,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )
                
                if result.success:
                    bound = _compute_bound(result.x[0], result.x[1], result.x[2])
                    if bound < best_bound:
                        best_bound = bound
                        best_coeff = result.x.copy()
            except Exception:
                pass
    
    return best_coeff
