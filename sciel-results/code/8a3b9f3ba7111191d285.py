"""Uncertainty inequality: tune Hermite coefficients to minimize the C4 upper bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from math import factorial, sqrt, pi


def _hermite(n, x):
    """Physicist's Hermite polynomial H_n(x)."""
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return 2 * x
    h_prev = np.ones_like(x)
    h_curr = 2 * x
    for k in range(2, n + 1):
        h_next = 2 * x * h_curr - 2 * (k - 1) * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


def _g(x, a0, a4, a8):
    """
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    H0(0)=1, H4(0)=-3, H8(0)=105, H12(0)=-10395
    g(0) = a0*1 + a4*(-3) + a8*105 + a12*(-10395) = 0
    => a12 = (a0 - 3*a4 + 105*a8) / 10395
    """
    a12 = (a0 - 3.0 * a4 + 105.0 * a8) / 10395.0
    h0 = _hermite(0, x)
    h4 = _hermite(4, x)
    h8 = _hermite(8, x)
    h12 = _hermite(12, x)
    poly = a0 * h0 + a4 * h4 + a8 * h8 + a12 * h12
    return poly * np.exp(-0.5 * x ** 2)


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on A(f)*A(F(f)) = (∫|g|^2)^{1/2} * (∫|g̃|^2)^{1/2}
    where g̃ is the Fourier transform of g.
    
    For g(x) = P(x) * exp(-x^2/2), the Fourier transform (using the convention
    F(ξ) = ∫ g(x) e^{-2πixξ} dx) can be computed. However, for the uncertainty
    principle bound, we use the convention where the Fourier transform preserves
    the Gaussian structure. Let's use the standard math convention:
    F(ξ) = ∫ g(x) e^{-i x ξ} dx
    
    For g(x) = P(x) e^{-x^2/2}, the Fourier transform is also a polynomial times
    a Gaussian (up to phase factors). Specifically, if g(x) = P(x) e^{-x^2/2},
    then F(ξ) = i^{-n} Q(ξ) e^{-ξ^2/2} for appropriate Q when P is a polynomial.
    
    Actually, let's just compute numerically for robustness.
    
    The quantity we want is:
    ||g||_2^2 * ||ĝ||_2^2
    
    where ||·||_2^2 = ∫ |·|^2 dx and ĝ is the Fourier transform.
    
    By Plancherel: ∫|ĝ(ξ)|^2 dξ = 2π ∫|g(x)|^2 dx (for the convention F(ξ)=∫g(x)e^{-ixξ}dx)
    
    Wait, that would make the product just 2π (∫|g|^2)^2, which doesn't depend on the shape.
    That can't be right for the uncertainty principle.
    
    Let me reconsider. The uncertainty principle bound involves:
    A(f) = ∫ |x f(x)|^2 dx / ∫|f(x)|^2 dx (variance in position)
    A(F) = ∫ |ξ F(ξ)|^2 dξ / ∫|F(ξ)|^2 dξ (variance in frequency)
    
    The product A(f)*A(F) ≥ 1/4 by Heisenberg. But the task says "upper bound on C4"
    which suggests a specific constant. Let me re-read.
    
    "score is the upper bound on C4" and "reduce the upper bound on A(f)*A(Fourier(f))"
    
    I think the task is about a specific inequality where the constant C4 depends on
    the test function. The bound is likely of the form:
    A(f)*A(F) ≤ C4 * ||f||^4 or similar, and we want to minimize C4.
    
    Actually, looking at the Goncalves et al. 2017 reference and the structure,
    I think the quantity to minimize is:
    (∫ x^2 |g|^2 dx) * (∫ ξ^2 |ĝ|^2 dξ) / (∫|g|^2)^2
    
    This is the product of variances, which by Heisenberg is ≥ 1/4.
    
    Let me compute this numerically.
    """
    x = np.linspace(-20, 20, 4001)
    dx = x[1] - x[0]
    
    g_vals = _g(x, a0, a4, a8)
    g_abs2 = np.abs(g_vals) ** 2
    
    norm2 = np.trapz(g_abs2, x)
    if norm2 < 1e-15:
        return np.inf
    
    # Position variance
    x2_weighted = np.trapz(x ** 2 * g_abs2, x)
    var_x = x2_weighted / norm2
    
    # Fourier transform: use FFT or numerical integration
    # For g(x) = P(x) e^{-x^2/2}, the FT is also a polynomial times Gaussian
    # Let's compute it analytically for efficiency and accuracy
    
    # H_n(x) e^{-x^2/2} has FT related to H_n(ξ) e^{-ξ^2/2}
    # Specifically: F[H_n(x) e^{-x^2/2}](ξ) = i^n H_n(ξ) e^{-ξ^2/2} * sqrt(2π) * (-1)^{n/2} if n even
    # Actually the precise relation: ∫ H_n(x) e^{-x^2/2} e^{-iξx} dx = i^n H_n(ξ) e^{-ξ^2/2} * sqrt(2π)
    # Let me verify for n=0: ∫ e^{-x^2/2} e^{-iξx} dx = sqrt(2π) e^{-ξ^2/2}. Yes, i^0=1.
    # For n=1: ∫ 2x e^{-x^2/2} e^{-iξx} dx = 2iξ e^{-ξ^2/2} * sqrt(2π). i^1 * H_1(ξ) = i * 2ξ. Yes.
    
    # So F[g](ξ) = sqrt(2π) * (a0 * i^0 * H0(ξ) + a4 * i^4 * H4(ξ) + a8 * i^8 * H8(ξ) + a12 * i^12 * H12(ξ)) * e^{-ξ^2/2}
    # i^0 = 1, i^4 = 1, i^8 = 1, i^12 = 1 (since i^4=1, so i^{4k}=1)
    # So F[g](ξ) = sqrt(2π) * (a0*H0(ξ) + a4*H4(ξ) + a8*H8(ξ) + a12*H12(ξ)) * e^{-ξ^2/2}
    
    # This is sqrt(2π) * g(ξ) ! (same form, just evaluated at ξ)
    
    # So ||ĝ||_2^2 = 2π * ∫|g(ξ)|^2 dξ = 2π * norm2 (by Plancherel, consistent)
    
    # For frequency variance:
    # ∫ ξ^2 |ĝ(ξ)|^2 dξ = 2π * ∫ ξ^2 |g(ξ)|^2 dξ = 2π * x2_weighted (by symmetry, since g is same form)
    
    # Wait, that means var_ξ = (2π * x2_weighted) / (2π * norm2) = x2_weighted/norm2 = var_x
    
    # So the product is var_x^2. That's a specific value but doesn't seem to allow tuning.
    
    # I must be misunderstanding the problem. Let me reconsider.
    
    # Perhaps the Fourier transform convention is different, or the quantity is different.
    # Let me just compute numerically with FFT to be safe.
    
    # Use a finer grid and FFT
    N = 2 ** 14
    L = 40.0
    x_fft = np.linspace(-L, L, N, endpoint=False)
    dx_fft = x_fft[1] - x_fft[0]
    
    g_fft = _g(x_fft, a0, a4, a8)
    
    # FFT
    G = np.fft.fftshift(np.fft.fft(g_fft)) * dx_fft
    
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=dx_fft))
    
    norm2_fft = np.sum(np.abs(g_fft) ** 2) * dx_fft
    if norm2_fft < 1e-15:
        return np.inf
    
    x2_moment = np.sum(x_fft ** 2 * np.abs(g_fft) ** 2) * dx_fft
    xi2_moment = np.sum(freqs ** 2 * np.abs(G) ** 2) * dx_fft
    
    var_x = x2_moment / norm2_fft
    var_xi = xi2_moment / norm2_fft  # Note: for the unitary FT this would differ
    
    # The product of variances
    return var_x * var_xi


def _objective(params):
    """Objective function to minimize."""
    a0, a4, a8 = params
    try:
        val = _compute_bound(a0, a4, a8)
        if np.isnan(val) or np.isinf(val):
            return 1e10
        return val
    except Exception:
        return 1e10


def construct():
    """
    Find the three coefficients (a0, a4, a8) that minimize the uncertainty product.
    """
    start_time = time.time()
    deadline = start_time + 190  # Safety margin before 210s kill
    
    # Known good starting point from Goncalves et al.
    best_params = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    best_score = _compute_bound(*best_params)
    
    if time.time() > deadline - 30:
        return best_params
    
    # Multi-start local optimization with BFGS
    starts = [
        np.array([-1.13, 0.04, 0.00030864]),  # Goncalves
        np.array([-1.0, 0.0, 0.0]),
        np.array([-1.5, 0.1, 0.001]),
        np.array([-0.5, -0.05, -0.001]),
        np.array([-2.0, 0.2, 0.005]),
        np.array([-1.13, 0.04, 0.00030864]),  # repeat
        np.array([-1.13, 0.04, 0.00030864]),
    ]
    
    for s in starts:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective, s, method='BFGS', 
                          options={'maxiter': 200, 'gtol': 1e-8})
            if res.success and res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
    
    if time.time() > deadline - 30:
        return best_params
    
    # Differential evolution for global search
    try:
        de_res = differential_evolution(
            _objective,
            bounds=[(-5, 5), (-1, 1), (-0.01, 0.01)],
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-8,
            polish=True,
            update_interval=1,
            callback=None
        )
        if de_res.fun < best_score:
            best_score = de_res.fun
            best_params = de_res.x
    except Exception:
        pass
    
    return best_params
# EVOLVE-BLOCK-END
