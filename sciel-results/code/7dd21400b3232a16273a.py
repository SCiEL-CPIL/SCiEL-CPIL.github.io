"""Combined program: analytic Hermite integrals + multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_START_TIME = time.time()
_DEADLINE = 170.0  # safety margin before 210s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _hermite_coeffs(c0, c4, c8):
    """Return all four coefficients given free ones, with g(0)=0 constraint."""
    # Probabilists' Hermite: H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    # Wait - let me use the correct convention. The seed uses specific values.
    # From the seed: coefficients [-113/100, 1/25, 1/3240]
    # H_k(0) for physicists' Hermite: H0(0)=1, H4(0)=-12, H8(0)=1680, H12(0)=-103950
    # For probabilists': He0(0)=1, He4(0)=-3, He8(0)=105, He12(0)=-10395
    # The seed values suggest probabilists' convention based on the magnitudes.
    h0_0 = 1.0
    h4_0 = -3.0
    h8_0 = 105.0
    h12_0 = -10395.0
    c12 = -(c0 * h0_0 + c4 * h4_0 + c8 * h8_0) / h12_0
    return c0, c4, c8, c12


def _compute_bound_analytic(c0, c4, c8):
    """
    Compute the upper bound using analytic Hermite polynomial properties.
    
    For f(x) = p(x) * exp(-x^2/2), the key quantities:
    - ||f||_2^2 = sum c_i^2 * ||H_i exp(-x^2/2)||_2^2 (by orthogonality)
    - Since FT preserves L2 norm (unitary), ||F(f)||_2 = ||f||_2
    - The uncertainty product bound involves ratios of these norms
    
    The "upper bound on C4" from Goncalves et al. likely involves:
    C4 = ||f||_1 * ||F(f)||_1 / (||f||_2^2) or similar
    
    Since |F(f)(xi)| = |f(xi)| (all phases are +1 for degrees 0,4,8,12),
    we have ||f||_1 = ||F(f)||_1, so the product is ||f||_1^2.
    
    But ||f||_1 cannot be computed analytically easily. Let's use a 
    high-accuracy numerical quadrature for the L1 norm, which is fast.
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Use Gauss-Hermite quadrature for high accuracy
    # ∫ g(x) exp(-x^2) dx = sum w_i g(x_i) for appropriate g
    # But we need ∫ |p(x) exp(-x^2/2)| dx, which is not a simple Gaussian integral
    
    # Use adaptive quadrature on a fine grid - this is fast enough
    x = np.linspace(-15, 15, 100001)
    dx = x[1] - x[0]
    
    # Probabilists' Hermite polynomials
    # He0 = 1
    # He4 = x^4 - 6x^2 + 3
    # He8 = x^8 - 28x^6 + 210x^4 - 420x^2 + 105
    # He12 = x^12 - 42x^10 + 1260x^8 - 16800x^6 + 113400x^4 - 302400x^2 + 10395
    
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2**2 - 6*x2 + 3
    H8 = x2**4 - 28*x2**3 + 210*x2**2 - 420*x2 + 105
    H12 = x2**6 - 42*x2**5 + 1260*x2**4 - 16800*x2**3 + 113400*x2**2 - 302400*x2 + 10395
    
    p = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    f = p * np.exp(-x2 / 2.0)
    
    # L1 norm of f
    A_f = np.sum(np.abs(f)) * dx
    
    # Since |F(f)| = |f| by the phase symmetry, A_F = A_f
    # But we need to verify the normalization. Let's compute FT numerically 
    # on a coarser grid to check, or just use the symmetry.
    
    # Actually, let's be more careful. The FT convention matters.
    # Using F(f)(xi) = ∫ f(x) exp(-2πi x xi) dx
    # For f(x) = H_n(x) exp(-x^2/2), the FT involves a Gaussian integral
    # with Hermite polynomial, which gives back a Hermite-Gaussian.
    
    # Let's compute numerically to be safe, but use FFT for speed
    N = len(x)
    F_f = np.fft.fft(f) * dx
    F_f = np.fft.fftshift(F_f)
    
    # Frequency grid
    freqs = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    freqs = np.fft.fftshift(freqs)
    dxi = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    
    A_F = np.sum(np.abs(F_f)) * dxi
    
    return A_f * A_F


def _compute_bound_fast(c0, c4, c8):
    """
    Faster computation using the key symmetry: for degrees 0,4,8,12 (all ≡0 mod 4),
    the Fourier transform phases are all +1, so |F(f)(xi)| = |f(xi)| up to 
    a constant factor depending on normalization.
    
    Let's compute ||f||_1 numerically and use the fact that the FT of 
    H_n exp(-x^2/2) has the same L1 norm (by the self-reciprocal property
    of the Gaussian and the phase being 1).
    
    Actually, the cleanest approach: compute ||f||_1 with high accuracy,
    and the bound is ||f||_1 * ||F(f)||_1. By the symmetry, these are 
    proportional. Let me just compute both numerically but efficiently.
    """
    c0, c4, c8, c12 = _hermite_coeffs(c0, c4, c8)
    
    # Use a moderate grid for speed during optimization
    x = np.linspace(-12, 12, 50001)
    dx = x[1] - x[0]
    
    x2 = x * x
    H0 = np.ones_like(x)
    H4 = x2**2 - 6*x2 + 3
    H8 = x2**4 - 28*x2**3 + 210*x2**2 - 420*x2 + 105
    H12 = x2**6 - 42*x2**5 + 1260*x2**4 - 16800*x2**3 + 113400*x2**2 - 302400*x2 + 10395
    
    p = c0 * H0 + c4 * H4 + c8 * H8 + c12 * H12
    f = p * np.exp(-x2 / 2.0)
    
    A_f = np.sum(np.abs(f)) * dx
    
    # By symmetry (all phases +1), A_F should equal A_f times a normalization factor
    # For the standard FT: F(f)(xi) = ∫ f(x) e^{-2πixxi} dx
    # The L1 norm of F(f) relates to L1 norm of f by Plancherel-like bounds
    # But since the function is self-reciprocal up to phase, let's just use A_f^2 
    # as a proxy if the normalization makes them equal.
    
    # To be safe and accurate, compute FT with FFT
    N = len(x)
    F_f = np.fft.fft(f) * dx
    F_f = np.fft.fftshift(F_f)
    freqs = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    freqs = np.fft.fftshift(freqs)
    dxi = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    
    A_F = np.sum(np.abs(F_f)) * dxi
    
    return A_f * A_F


def construct():
    """
    Find the three free coefficients (c0, c4, c8) for the Hermite combination
    to minimize the upper bound on A(f)*A(Fourier(f)).
    
    Strategy:
    1. Start from the seed (Goncalves et al. 2017) values
    2. Use local optimization (Nelder-Mead) from the seed
    3. If time permits, try a few random restarts with local refinement
    4. Return the best coefficients found
    """
    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Evaluate seed
    best_val = _compute_bound_fast(*seed)
    best_params = seed.copy()
    
    # Local optimization from seed
    if _time_left() > 30:
        try:
            res = minimize(
                _compute_bound_fast,
                seed,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Try a few random perturbations around the best point
    if _time_left() > 40:
        rng = np.random.default_rng(42)
        for i in range(5):
            if _time_left() < 20:
                break
            try:
                # Small perturbations
                scale = 0.1 / (i + 1)
                perturb = best_params + rng.normal(0, scale, size=3)
                res = minimize(
                    _compute_bound_fast,
                    perturb,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                if res.fun < best_val:
                    best_val = res.fun
                    best_params = res.x
            except Exception:
                pass
    
    # Final refinement with tighter tolerances
    if _time_left() > 15:
        try:
            res_final = minimize(
                _compute_bound_fast,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if res_final.fun < best_val:
                best_val = res_final.fun
                best_params = res_final.x
        except Exception:
            pass
    
    return np.array(best_params)
# EVOLVE-BLOCK-END
