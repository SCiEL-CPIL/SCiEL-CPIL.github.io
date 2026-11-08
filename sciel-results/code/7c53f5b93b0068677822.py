"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

def _hermite_coeffs(a0, a4, a8):
    """
    Given coefficients for H0, H4, H8, compute the H12 coefficient
    so that g(0) = 0.
    
    The test function is g(x) = exp(-x^2/2) * (a0*H0 + a4*H4 + a8*H8 + a12*H12)
    where Hn are probabilists' Hermite polynomials (He_n) or physicists' (H_n)?
    
    The task says "Hermite-polynomial times Gaussian". Typically in uncertainty
    principle contexts, we use physicists' Hermite polynomials H_n with the
    Gaussian exp(-x^2/2), or probabilists' He_n with exp(-x^2/2).
    
    Let's assume physicists' Hermite polynomials H_n(x) with the standard
    Gaussian weight exp(-x^2/2). The Hermite functions are psi_n(x) = 
    (1/sqrt(2^n n! sqrt(pi))) * H_n(x) * exp(-x^2/2).
    
    But the task says "Hermite-polynomial times Gaussian test function", so
    g(x) = P(x) * exp(-x^2/2) where P is a polynomial.
    
    For g(0) = 0: P(0) * exp(0) = 0 => P(0) = 0.
    
    Physicists' Hermite polynomials:
    H0(x) = 1
    H4(x) = 16x^4 - 48x^2 + 12
    H8(x) = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1680
    
    So P(0) = a0*1 + a4*12 + a8*1680 + a12*H8(0)... wait, H12(0).
    
    Actually, let me reconsider. The problem states:
    "Three free coefficients multiply H0, H4, H8; the notebook derives the H12 
    coefficient to impose P(0)=0."
    
    So P(x) = a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)
    and P(0) = 0 determines a12.
    
    For physicists' Hermite polynomials:
    H0(0) = 1
    H4(0) = 12
    H8(0) = 1680
    H12(0) = ?
    
    H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5728704x^6 + 15482880x^4 - 20321280x^2 + 967680
    
    H12(0) = 967680
    
    So a12 = -(a0*1 + a4*12 + a8*1680) / 967680
    """
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 967680.0
    a12 = -(a0 * H0_0 + a4 * H4_0 + a8 * H8_0) / H12_0
    return np.array([a0, a4, a8, a12])

def _eval_hermite(x, coeffs):
    """
    Evaluate the polynomial part P(x) = sum c_i * H_i(x) for i in [0,4,8,12].
    """
    a0, a4, a8, a12 = coeffs
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 10752x^4 - 10752x^2 + 1680
    # H12(x) = 4096x^12 - 98304x^10 + 1032192x^8 - 5728704x^6 + 15482880x^4 - 20321280x^2 + 967680
    H0 = np.ones_like(x)
    H4 = 16*x**4 - 48*x**2 + 12
    H8 = 256*x**8 - 3072*x**6 + 10752*x**4 - 10752*x**2 + 1680
    H12 = (4096*x**12 - 98304*x**10 + 1032192*x**8 - 5728704*x**6 
           + 15482880*x**4 - 20321280*x**2 + 967680)
    return a0*H0 + a4*H4 + a8*H8 + a12*H12

def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on A(f)*A(Fourier(f)) for the test function
    g(x) = P(x) * exp(-x^2/2) where P is the Hermite polynomial combination.
    
    The uncertainty inequality bound involves the product of the L1 norms
    or similar quantities. For a function g, A(g) might be defined as
    the integral of |g(x)| or some related quantity.
    
    Actually, in the context of uncertainty principles, the bound is often
    on ||g||_1 * ||hat{g}||_1 or similar. Let me think about what C4 refers to.
    
    The task says "score is the upper bound on C4". This likely refers to a
    specific constant in an uncertainty inequality. 
    
    For a function g(x) = P(x) exp(-x^2/2), we can compute:
    - ||g||_1 = integral |g(x)| dx
    - ||hat{g}||_1 = integral |hat{g}(xi)| dxi
    
    The Fourier transform of g(x) = P(x) exp(-x^2/2) can be computed analytically
    using the fact that the Fourier transform of x^n exp(-x^2/2) is related to
    derivatives of the Gaussian Fourier transform.
    
    The Fourier transform of exp(-x^2/2) is sqrt(2*pi) * exp(-xi^2/2).
    
    For P(x) = sum c_k x^k, the Fourier transform involves derivatives of the
    Gaussian. Specifically, F[x^n f(x)] = i^n d^n/dxi^n F[f](xi).
    
    So for g(x) = P(x) exp(-x^2/2), hat{g}(xi) = sqrt(2*pi) * Q(xi) * exp(-xi^2/2)
    where Q is a polynomial related to P.
    
    Let me compute this more carefully. Let's use the convention:
    hat{f}(xi) = integral f(x) exp(-2*pi*i*x*xi) dx
    
    Then F[exp(-x^2/2)] = sqrt(2*pi) * exp(-2*pi^2*xi^2)
    
    Hmm, this depends on the Fourier transform convention. Let me use the
    angular frequency convention: hat{f}(omega) = integral f(x) exp(-i*omega*x) dx
    
    Then F[exp(-x^2/2)] = sqrt(2*pi) * exp(-omega^2/2)
    
    And F[x^n exp(-x^2/2)] = i^n d^n/domega^n [sqrt(2*pi) * exp(-omega^2/2)]
    
    For a polynomial P(x) = sum c_k x^k, we have:
    hat{g}(omega) = sum c_k F[x^k exp(-x^2/2)] = i * sqrt(2*pi) * R(omega) * exp(-omega^2/2)
    
    where R is some polynomial.
    
    Actually, let me just numerically compute the bound. The bound on
    A(f)*A(Fourier(f)) is likely ||f||_1 * ||hat{f}||_1 or similar.
    
    Let me compute ||g||_1 and ||hat{g}||_1 numerically.
    """
    coeffs = _hermite_coeffs(a0, a4, a8)
    
    # Compute ||g||_1 = integral |g(x)| dx
    # g(x) = P(x) * exp(-x^2/2)
    x = np.linspace(-15, 15, 10000)
    P = _eval_hermite(x, coeffs)
    g = P * np.exp(-x**2 / 2)
    norm_g_1 = np.trapz(np.abs(g), x)
    
    # Compute hat{g}(omega) = integral g(x) exp(-i*omega*x) dx
    # Use the analytic formula or numerical FFT
    # Let's use the analytic approach: hat{g}(omega) = sqrt(2*pi) * R(omega) * exp(-omega^2/2)
    
    # For P(x) = sum c_k H_k(x), we need to express H_k(x) in terms of x^j.
    # This is complex. Let me just use numerical integration for the Fourier transform.
    
    omega = np.linspace(-15, 15, 10000)
    # Compute hat{g}(omega) = integral g(x) exp(-i*omega*x) dx
    # This is a 2D integral, which is expensive. Let me use the fact that
    # for g(x) = P(x) exp(-x^2/2), the Fourier transform is also a polynomial
    # times Gaussian.
    
    # Let me compute the Fourier transform analytically.
    # F[x^n exp(-x^2/2)] = i^n d^n/domega^n [sqrt(2*pi) exp(-omega^2/2)]
    
    # d/domega [exp(-omega^2/2)] = -omega * exp(-omega^2/2)
    # d^2/domega^2 [exp(-omega^2/2)] = (omega^2 - 1) * exp(-omega^2/2)
    # d^3/domega^3 [exp(-omega^2/2)] = -(omega^3 - 3omega) * exp(-omega^2/2)
    # d^4/domega^4 [exp(-omega^2/2)] = (omega^4 - 6omega^2 + 3) * exp(-omega^2/2)
    
    # So F[x^n exp(-x^2/2)] = i^n sqrt(2*pi) * He_n(omega) * exp(-omega^2/2)
    # where He_n are probabilists' Hermite polynomials (up to sign conventions).
    
    # Actually, let me just use numerical FFT on a fine grid.
    N = 1024
    x_grid = np.linspace(-15, 15, N)
    g_grid = _eval_hermite(x_grid, coeffs) * np.exp(-x_grid**2 / 2)
    
    # Compute Fourier transform using FFT
    # hat{g}(omega) = integral g(x) exp(-i*omega*x) dx
    # Using FFT: hat{g}(omega_k) ≈ dx * sum g(x_j) exp(-i*omega_k*x_j)
    # where omega_k = 2*pi*k/(N*dx)
    
    dx = x_grid[1] - x_grid[0]
    G = np.fft.fftshift(np.fft.fft(g_grid)) * dx
    omega_grid = 2*np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    norm_g_hat_1 = np.trapz(np.abs(G), omega_grid)
    
    return norm_g_1 * norm_g_hat_1

def construct():
    """
    Find the coefficients a0, a4, a8 that minimize the upper bound on
    A(f)*A(Fourier(f)).
    """
    start_time = time.time()
    deadline = start_time + 180  # 180 second limit
    
    # Initial best from the seed program
    best_coeffs = np.array([-113 / 100, 1 / 25, 1 / 3240])
    best_bound = _compute_bound(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    
    # Use differential evolution for global optimization
    bounds = [(-10, 10), (-10, 10), (-10, 10)]
    
    def objective(x):
        return _compute_bound(x[0], x[1], x[2])
    
    try:
        result = differential_evolution(
            objective, bounds, 
            maxiter=100, popsize=15,
            tol=1e-8, seed=42
        )
        
        if time.time() < deadline - 10:
            # Refine with local optimization
            result2 = minimize(
                objective, result.x, 
                method='L-BFGS-B', bounds=bounds
            )
            if result2.fun < best_bound:
                best_coeffs = result2.x
                best_bound = result2.fun
        else:
            if result.fun < best_bound:
                best_coeffs = result.x
                best_bound = result.fun
    
    except Exception:
        pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
