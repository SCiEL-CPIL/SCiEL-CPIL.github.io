"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
import scipy.special as sp
from scipy.integrate import quad


def _hermite_polynomial(n, x):
    """Compute the probabilists' Hermite polynomial Hn(x) using scipy."""
    return sp.eval_hermite(n, x)


def _construct_g(a0, a4, a8):
    """
    Given coefficients for H0, H4, H8, compute the H12 coefficient such that g(0) = 0.
    g(x) = (a0*H0(x) + a4*H4(x) + a8*H8(x) + a12*H12(x)) * exp(-x^2/2)
    We need g(0) = 0.
    H0(0) = 1
    H4(0) = -3
    H8(0) = 105
    H12(0) = -10395
    So: a0*1 + a4*(-3) + a8*105 + a12*(-10395) = 0
    => a12 = (a0 - 3*a4 + 105*a8) / 10395
    """
    a12 = (a0 - 3 * a4 + 105 * a8) / 10395.0
    return np.array([a0, a4, a8, a12])


def _g(x, coeffs):
    """Evaluate the test function g(x) = sum(ai * Hi(x)) * exp(-x^2/2)."""
    a0, a4, a8, a12 = coeffs
    val = (a0 * _hermite_polynomial(0, x) +
           a4 * _hermite_polynomial(4, x) +
           a8 * _hermite_polynomial(8, x) +
           a12 * _hermite_polynomial(12, x)) * np.exp(-x**2 / 2)
    return val


def _compute_upper_bound_C4(coeffs):
    """
    Compute the upper bound on C4 for the given Hermite combination.
    
    The uncertainty inequality gives:
    A(f) * A(Fourier(f)) >= C4 * ||f||_4^2
    
    For our test function g(x), we compute the upper bound by:
    1. Computing ||g||_4^4 = ∫|g(x)|^4 dx
    2. Computing ||ĝ||_4^4 = ∫|ĝ(ξ)|^4 dξ (using Parseval/Plancherel)
    3. The product A(g)*A(ĝ) where A(h) = (∫|h|^4)^(1/4)
    
    Actually, for the uncertainty inequality in the form:
    (∫|f|^4 dx)^{1/2} (∫|F(f)|^4 dξ)^{1/2} >= C4 * (∫|f|^2 dx)
    
    Wait, let me reconsider. The standard uncertainty inequality for L^4:
    ||f||_4 * ||F(f)||_4 >= C4 * ||f||_2
    
    where F is the Fourier transform with appropriate normalization.
    
    Let's use the convention: F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx
    Then ||F(f)||_2 = ||f||_2 (Plancherel)
    
    The bound is: ||f||_4 * ||F(f)||_4 >= C4 * ||f||_2
    
    So C4_upper = ||f||_4 * ||F(f)||_4 / ||f||_2
    
    We want to MINIMIZE this upper bound to get the tightest bound.
    
    For g(x) = (polynomial in H's) * exp(-x^2/2):
    - ||g||_2^2 = ∫ |g(x)|^2 dx
    - ||g||_4^4 = ∫ |g(x)|^4 dx
    - ||F(g)||_4^4 = ∫ |F(g)(ξ)|^4 dξ
    
    By Plancherel: ||F(g)||_2^2 = ||g||_2^2
    But we need the L^4 norm of the Fourier transform.
    
    For a Gaussian times polynomial, the Fourier transform is also a Gaussian times polynomial.
    Specifically, if g(x) = P(x) * exp(-x^2/2), then
    F(g)(ξ) = P~(ξ) * exp(-2π^2 ξ^2) (up to constants depending on convention)
    
    Let me use the convention: F(f)(ξ) = ∫ f(x) e^{-i x ξ} dx (angular frequency)
    Then F(exp(-x^2/2))(ξ) = sqrt(2π) * exp(-ξ^2/4)
    
    Actually, let me just compute numerically.
    """
    a0, a4, a8, a12 = coeffs
    
    # Define g(x)
    def g_func(x):
        val = (a0 * sp.eval_hermite(0, x) +
               a4 * sp.eval_hermite(4, x) +
               a8 * sp.eval_hermite(8, x) +
               a12 * sp.eval_hermite(12, x)) * np.exp(-x**2 / 2)
        return val
    
    # Compute ||g||_2^2 = ∫ |g(x)|^2 dx
    norm2_sq, _ = quad(lambda x: g_func(x)**2, -np.inf, np.inf, limit=200)
    norm2 = np.sqrt(norm2_sq)
    
    # Compute ||g||_4^4 = ∫ |g(x)|^4 dx
    norm4_4, _ = quad(lambda x: g_func(x)**4, -np.inf, np.inf, limit=200)
    norm4 = norm4_4 ** 0.25
    
    # For the Fourier transform, we need to compute ||F(g)||_4
    # Using the convention F(f)(ξ) = ∫ f(x) e^{-ixξ} dx
    # The Fourier transform of exp(-x^2/2) is sqrt(2π) * exp(-ξ^2/4)
    # The Fourier transform of x^n * exp(-x^2/2) can be computed.
    
    # Let's compute F(g)(ξ) numerically for the L^4 norm
    def F_g_func(xi):
        # Compute ∫ g(x) e^{-ixξ} dx numerically
        def integrand(x):
            return g_func(x) * np.cos(x * xi)  # Real part; g is real and even
        # g is real and even (Hermite polynomials of even degree times even Gaussian),
        # so F(g) is real and even
        val, _ = quad(integrand, -np.inf, np.inf, limit=200)
        return val
    
    # This will be slow. Let's use the fact that for Gaussian times polynomial,
    # the Fourier transform is also Gaussian times polynomial.
    
    # Alternative: use the property that for g(x) = Σ c_n H_n(x) e^{-x^2/2},
    # F(g)(ξ) = sqrt(2π) * Σ c_n (-i)^n H_n(ξ/2) e^{-ξ^2/4}
    # Wait, this depends on the exact convention.
    
    # Let me use a different approach. Since we're dealing with even functions,
    # let's compute the Fourier cosine transform.
    
    # Actually, for the purpose of this problem, let's use a simpler approach:
    # We know that the Fourier transform of H_n(x) * e^{-x^2/2} is (-i)^n * sqrt(2π) * H_n(ξ/2) * e^{-ξ^2/4}
    # (with the convention F(f)(ξ) = ∫ f(x) e^{-ixξ} dx)
    
    # Let's verify: H_0(x) e^{-x^2/2} -> sqrt(2π) e^{-ξ^2/4}
    # H_2(x) e^{-x^2/2} = (x^2-1) e^{-x^2/2}
    # F((x^2-1) e^{-x^2/2}) = ?
    # F(x^2 e^{-x^2/2}) = -d^2/dξ^2 [sqrt(2π) e^{-ξ^2/4}] = sqrt(2π) (ξ^2/4 - 1/2) e^{-ξ^2/4}... 
    # Hmm, this is getting complicated. Let me just compute numerically.
    
    # For efficiency, let's sample on a grid and use FFT
    # But the function decays as exp(-x^2/2), so we can truncate.
    
    N = 4096
    L = 20  # Integration range [-L, L]
    x = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    
    g_vals = np.array([g_func(xi) for xi in x])
    
    # Compute Fourier transform using FFT
    # F(g)(ξ) ≈ ∫_{-L}^{L} g(x) e^{-ixξ} dx
    # Using FFT: if we have samples g(x_k) at x_k = -L + k*dx,
    # then F(g)(ξ_j) ≈ dx * Σ g(x_k) e^{-i x_k ξ_j}
    # where ξ_j = 2π j / (N dx) for j = 0, ..., N-1
    
    G = np.fft.fft(g_vals) * dx
    # Shift to center zero frequency
    G = np.fft.fftshift(G)
    
    # Frequency grid
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx))
    
    # ||F(g)||_4^4 = ∫ |F(g)(ξ)|^4 dξ ≈ Σ |G_j|^4 * dξ
    dξ = freqs[1] - freqs[0]
    F_norm4_4 = np.sum(np.abs(G)**4) * dξ
    F_norm4 = F_norm4_4 ** 0.25
    
    # Upper bound on C4
    if norm2 > 0:
        C4_bound = norm4 * F_norm4 / norm2
    else:
        C4_bound = np.inf
    
    return C4_bound


def construct():
    """
    Find three coefficients (a0, a4, a8) that minimize the upper bound on C4.
    The H12 coefficient is determined by the constraint g(0) = 0.
    """
    start_time = time.time()
    deadline = start_time + 180  # Must finish within 180s
    
    # Initial guess: the seed program values
    a0_init = -113 / 100
    a4_init = 1 / 25
    a8_init = 1 / 3240
    
    def objective(params):
        a0, a4, a8 = params
        coeffs = _construct_g(a0, a4, a8)
        return _compute_upper_bound_C4(coeffs)
    
    # Try Powell method
    try:
        result = opt.minimize(objective, [a0_init, a4_init, a8_init],
                             method='Powell',
                             options={'maxiter': 50, 'ftol': 1e-10, 'xtol': 1e-10})
        best_params = result.x
        best_score = result.fun
    except Exception:
        best_params = np.array([a0_init, a4_init, a8_init])
        best_score = objective(best_params)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        # Return the best we have
        return np.array([best_params[0], best_params[1], best_params[2]])
    
    # Try Nelder-Mead as well
    try:
        result2 = opt.minimize(objective, [a0_init, a4_init, a8_init],
                              method='Nelder-Mead',
                              options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
        if result2.fun < best_score:
            best_params = result2.x
            best_score = result2.fun
    except Exception:
        pass
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return np.array([best_params[0], best_params[1], best_params[2]])
    
    # Try a grid search in a neighborhood
    try:
        # Scale the search based on the current best
        scale = max(1.0, np.abs(best_params).max())
        grid_range = 2 * scale
        
        # Coarse grid
        n_grid = 10
        a0_grid = np.linspace(best_params[0] - grid_range, best_params[0] + grid_range, n_grid)
        a4_grid = np.linspace(best_params[1] - grid_range, best_params[1] + grid_range, n_grid)
        a8_grid = np.linspace(best_params[2] - grid_range, best_params[2] + grid_range, n_grid)
        
        for a0 in a0_grid:
            for a4 in a4_grid:
                for a8 in a8_grid:
                    if time.time() > deadline - 30:
                        break
                    score = objective([a0, a4, a8])
                    if score < best_score:
                        best_score = score
                        best_params = np.array([a0, a4, a8])
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
    except Exception:
        pass
    
    return np.array([best_params[0], best_params[1], best_params[2]])
# EVOLVE-BLOCK-END
