"""Rewritten program for the uncertainty inequality task."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
import time
import itertools


def _hermite_h(n, x):
    """Compute physicists' Hermite polynomial H_n(x) using recurrence."""
    if n == 0:
        return np.ones_like(x, dtype=float)
    if n == 1:
        return 2.0 * x
    h0 = np.ones_like(x, dtype=float)
    h1 = 2.0 * x
    for k in range(2, n + 1):
        h2 = 2.0 * x * h1 - 2.0 * (k - 1) * h0
        h0, h1 = h1, h2
    return h1


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 for the given Hermite coefficients.
    
    The test function is g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    where a12 is chosen so that g(0) = 0.
    
    The bound is A(f)*A(F(f)) where A is the area under |.| or similar.
    For Fourier-invariant functions, we use the Wigner function approach.
    
    The key insight: for g(x) = sum c_k H_k(x) e^{-x^2/2}, the Wigner function
    in phase space has a specific structure. The uncertainty bound relates to
    the L1 norm of the Wigner function and its Fourier transform.
    
    For this problem, the bound C4 is computed from the moments of |g|^2 and
    |F(g)|^2, or more precisely from the L1 norms.
    
    Let's use the direct numerical approach:
    - Compute g(x) on a grid
    - Compute F(g) via FFT or analytically
    - Compute the bound as product of appropriate norms
    
    Actually, for the uncertainty principle, the bound is:
    C4 = (∫|g|^2 dx) * (∫|F(g)|^2 dξ) / (some normalization)
    
    But the task says "upper bound on A(f)*A(Fourier(f))" where A likely means
    the area (L1 norm) or L2 norm.
    
    Looking at the seed program returning [-113/100, 1/25, 1/3240], and the
    score being the raw value (lower is better), with 1.0 being the seed score,
    the bound value for the seed is some specific number.
    
    Let me think about this more carefully. The "uncertainty inequality" with
    Hermite polynomials and Gaussian... The bound A(f)*A(F(f)) where A is likely
    the L1 norm (area under the absolute value curve).
    
    For a function g(x) = p(x) e^{-x^2/2} where p is a polynomial:
    - F(g)(ξ) can be computed analytically using the fact that F(H_k e^{-x^2/2})
      = i^k H_k(ξ) e^{-ξ^2/2} (up to constants depending on convention)
    
    With the convention F(f)(ξ) = ∫ f(x) e^{-2πixξ} dx:
    F(H_k(x) e^{-x^2/2})(ξ) = i^k H_k(ξ) e^{-ξ^2/2} * (some factor)
    
    Actually, with F(f)(ξ) = ∫ f(x) e^{-i x ξ} dx (angular frequency):
    F(e^{-x^2/2})(ξ) = √(2π) e^{-ξ^2/4}
    This doesn't preserve the form nicely.
    
    Let's use the convention where the Gaussian is e^{-x^2/4} or adjust.
    
    Actually, the standard result: if g(x) = e^{-x^2/2}, then F(g)(ξ) = √(2π) e^{-ξ^2/4}.
    This isn't invariant.
    
    For Fourier invariance, we need the convention F(f)(ξ) = (1/√(2π)) ∫ f(x) e^{-ixξ} dx.
    Then F(e^{-x^2/2})(ξ) = e^{-ξ^2/2}, which IS invariant.
    
    And F(H_k(x) e^{-x^2/2})(ξ) = i^k H_k(ξ) e^{-ξ^2/2}.
    
    So with this convention, if g(x) = Σ c_k H_k(x) e^{-x^2/2}, then
    F(g)(ξ) = Σ c_k i^k H_k(ξ) e^{-ξ^2/2}.
    
    Now, A(f) is likely the L1 norm: A(f) = ∫ |f(x)| dx.
    
    The bound is A(g) * A(F(g)).
    
    Since F(g)(ξ) = Σ c_k i^k H_k(ξ) e^{-ξ^2/2}, and |i^k| = 1, we have
    |F(g)(ξ)| = |Σ c_k i^k H_k(ξ) e^{-ξ^2/2}| = e^{-ξ^2/2} |Σ c_k i^k H_k(ξ)|.
    
    And |g(x)| = e^{-x^2/2} |Σ c_k H_k(x)|.
    
    So A(g) = ∫ e^{-x^2/2} |Σ c_k H_k(x)| dx
    A(F(g)) = ∫ e^{-ξ^2/2} |Σ c_k i^k H_k(ξ)| dξ
    
    Since the integral is over all reals and the weight is symmetric, and
    |Σ c_k i^k H_k(ξ)| vs |Σ c_k H_k(ξ)|, these are different in general.
    
    But wait - the problem says "Fourier-invariant Hermite-polynomial times Gaussian".
    This might mean that the combination is chosen to be Fourier-invariant, or it
    might just mean the basis functions are Fourier-invariant in form.
    
    Let me just compute numerically:
    - A(g) = ∫|g(x)|dx
    - A(F(g)) = ∫|F(g)(ξ)|dξ
    
    Using the analytic Fourier transform: F(g)(ξ) = Σ c_k i^k H_k(ξ) e^{-ξ^2/2}
    
    The constraint g(0) = 0 means Σ c_k H_k(0) = 0.
    H_0(0) = 1, H_4(0) = -12, H_8(0) = 1008, H_12(0) = -103968.
    
    So: a0*1 + a4*(-12) + a8*1008 + a12*(-103968) = 0
    => a12 = (a0 - 12*a4 + 1008*a8) / 103968
    
    Let me compute the bound numerically.
    """
    # a12 from g(0) = 0
    a12 = (a0 - 12.0 * a4 + 1008.0 * a8) / 103968.0
    
    coeffs = [a0, a4, a8, a12]
    degrees = [0, 4, 8, 12]
    
    # Numerical integration over a sufficiently large domain
    x = np.linspace(-20, 20, 20001)
    dx = x[1] - x[0]
    
    # Compute g(x)
    g = np.zeros_like(x)
    for c, d in zip(coeffs, degrees):
        g += c * _hermite_h(d, x) * np.exp(-x**2 / 2.0)
    
    A_g = np.trapz(np.abs(g), x)
    
    # Compute F(g)(ξ) analytically: F(g)(ξ) = Σ c_k i^k H_k(ξ) e^{-ξ^2/2}
    Fg = np.zeros_like(x)  # reuse x as ξ
    for c, d in zip(coeffs, degrees):
        Fg += c * (1j**d) * _hermite_h(d, x) * np.exp(-x**2 / 2.0)
    
    A_Fg = np.trapz(np.abs(Fg), x)
    
    return A_g * A_Fg


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the upper bound A(g)*A(F(g))
    subject to g(0) = 0 (which determines a12).
    """
    t_start = time.time()
    deadline = 180.0  # seconds, with safety margin from 210s kill
    
    # Seed values
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_val = _compute_bound(seed[0], seed[1], seed[2])
    best_x = seed.copy()
    
    def objective(v):
        return _compute_bound(v[0], v[1], v[2])
    
    # Multi-start BFGS with a grid of initial points
    # Wider grid based on findings
    grid_a0 = np.linspace(-1.5, 1.5, 20)
    grid_a4 = np.linspace(-0.8, 0.8, 15)
    grid_a8 = np.linspace(-0.05, 0.05, 10)
    
    n_starts = len(grid_a0) * len(grid_a4) * len(grid_a8)
    n_done = 0
    
    for a0_init in grid_a0:
        if time.time() - t_start > deadline - 30:
            break
        for a4_init in grid_a4:
            if time.time() - t_start > deadline - 30:
                break
            for a8_init in grid_a8:
                if time.time() - t_start > deadline - 10:
                    break
                x0 = np.array([a0_init, a4_init, a8_init])
                try:
                    res = minimize(objective, x0, method='BFGS', 
                                   options={'gtol': 1e-12, 'maxiter': 500})
                    n_done += 1
                    if res.success and res.fun < best_val:
                        best_val = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
    
    # Additional local refinement around best point
    if time.time() - t_start < deadline - 5:
        try:
            res = minimize(objective, best_x, method='BFGS',
                           options={'gtol': 1e-14, 'maxiter': 1000})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_x = res.x.copy()
        except Exception:
            pass
        
        # Try Nelder-Mead as a final polish
        if time.time() - t_start < deadline - 5:
            try:
                res = minimize(objective, best_x, method='Nelder-Mead',
                               options={'xatol': 1e-12, 'fatol': 1e-12, 'maxiter': 5000})
                if res.success and res.fun < best_val:
                    best_val = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
    
    return best_x

# EVOLVE-BLOCK-END
