"""Optimize Hermite coefficients to minimize the C4 upper bound on uncertainty product."""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import factorial


def _H_at_0(deg):
    """Compute H_n(0) for probabilists' Hermite polynomials."""
    if deg % 2 != 0:
        return 0.0
    j = deg // 2
    return ((-1) ** j) * float(factorial(deg, exact=True)) / float(factorial(j, exact=True))


# Precompute values at 0 for degrees 0, 4, 8, 12
_H0_0 = _H_at_0(0)    # 1
_H4_0 = _H_at_0(4)    # 12
_H8_0 = _H_at_0(8)    # -420
_H12_0 = _H_at_0(12)  # 11880

_DEGS = np.array([0, 4, 8, 12])


def _compute_bound(a0, a4, a8):
    """
    Compute the upper bound on C4 = A(f) * A(Fourier(f)).
    
    For g(x) = sum_k a_k H_k(x) e^{-x^2/2}:
    - ||g||_2^2 = sum_k a_k^2 k!  (orthogonality of probabilists' Hermite)
    - The Fourier transform: F{H_k e^{-x^2/2}} = i^k H_k e^{-x^2/2}
    - So ||ĝ||_2^2 = ||g||_2^2
    - A(g) = ∫|g|dx / ||g||_2
    - A(ĝ) = ∫|ĝ|dx / ||ĝ||_2
    
    Since |i^k| = 1, the structure of |ĝ| relates to |g| but with phase factors.
    However, for the L¹ norm, we need to account for the actual absolute values.
    
    The bound C4 = A(g) * A(ĝ) = [∫|g|dx / ||g||_2] * [∫|ĝ|dx / ||ĝ||_2]
    
    Since ||ĝ||_2 = ||g||_2, we have C4 = [∫|g|dx * ∫|ĝ|dx] / ||g||_2^2
    
    For the Fourier transform of the Hermite combination, we can compute
    ∫|ĝ|dx numerically or use the fact that for even combinations, the
    transform preserves the even structure.
    
    Actually, let me reconsider. The Fourier transform in the convention where
    F{e^{-x^2/2}} = e^{-ξ^2/2} (unitary) gives:
    F{H_k(x) e^{-x^2/2}} = i^k H_k(ξ) e^{-ξ^2/2}
    
    So ĝ(ξ) = sum_k a_k i^k H_k(ξ) e^{-ξ^2/2}
    
    For even k, i^k is real: i^0=1, i^4=1, i^8=1, i^12=1
    So for our even-degree combination, ĝ(ξ) = g(ξ) (same function!)
    
    Therefore ∫|ĝ|dx = ∫|g|dx, and C4 = [∫|g|dx]^2 / ||g||_2^2
    """
    # Compute a12 from g(0) = 0
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    coeffs = np.array([a0, a4, a8, a12])
    
    # Compute ||g||_2^2 = sum a_k^2 k!
    norms_sq = np.sum(coeffs ** 2 * np.array([float(factorial(d, exact=True)) for d in _DEGS]))
    if norms_sq <= 1e-15:
        return 1e12
    
    # Compute ∫|g(x)|dx using numerical quadrature
    # g(x) is even, so integrate from 0 to inf and multiply by 2
    # For large x, g(x) ~ polynomial * e^{-x^2/2}, so converges fast
    
    def g_abs(x):
        # Evaluate the Hermite polynomial combination
        val = (a0 * _eval_H0(x) + 
               a4 * _eval_H4(x) + 
               a8 * _eval_H8(x) + 
               a12 * _eval_H12(x)) * np.exp(-x**2 / 2.0)
        return abs(val)
    
    # Use adaptive quadrature
    try:
        from scipy.integrate import quad
        integral, _ = quad(g_abs, 0, 15, limit=200, epsabs=1e-10, epsrel=1e-10)
        integral *= 2.0  # even function
    except Exception:
        # Fallback: trapezoidal rule
        xs = np.linspace(0, 15, 5000)
        vals = np.array([g_abs(x) for x in xs])
        integral = 2.0 * np.trapz(vals, xs)
    
    bound = (integral ** 2) / norms_sq
    return bound


def _eval_H0(x):
    return 1.0


def _eval_H4(x):
    return x**4 - 6*x**2 + 3


def _eval_H8(x):
    return x**8 - 28*x**6 + 210*x**4 - 420*x**2 + 105


def _eval_H12(x):
    return (x**12 - 66*x**10 + 1260*x**8 - 13860*x**6 
            + 55440*x**4 - 66528*x**2 + 10395)


def construct():
    deadline = time.time() + 195.0  # Safety margin under 210s
    
    # Seed values
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best = seed.copy()
    best_val = _compute_bound(seed[0], seed[1], seed[2])
    
    def update(x, v):
        nonlocal best, best_val
        if np.all(np.isfinite(x)) and np.isfinite(v) and v < best_val:
            best_val = v
            best = np.array(x, dtype=float)
    
    # Stage 1: Differential evolution for global search
    try:
        de_res = differential_evolution(
            lambda x: _compute_bound(x[0], x[1], x[2]),
            bounds=[(-5.0, 5.0), (-0.5, 0.5), (-0.1, 0.1)],
            maxiter=200,
            popsize=15,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=42,
            init='latinhypercube',
            polish=True,
            maxfev=3000,
            time_limit=max(10.0, deadline - time.time() - 50.0),
            update_interval=1,
        )
        update(de_res.x, de_res.fun)
    except Exception:
        pass
    
    # Stage 2: Local refinement from multiple starting points
    rng = np.random.default_rng(123)
    starts = [best.copy()]
    
    # Add structured starts around the seed
    for scale in [0.5, 1.0, 2.0]:
        starts.append(seed + scale * 0.1 * rng.standard_normal(3))
    
    # Add random starts
    for _ in range(10):
        starts.append(rng.uniform([-3.0, -0.3, -0.05], [3.0, 0.3, 0.05]))
    
    for s in starts:
        if time.time() > deadline - 20.0:
            break
        try:
            res = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                s,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 5000, 'adaptive': True}
            )
            update(res.x, res.fun)
            
            # Second-order polish
            if time.time() < deadline - 15.0:
                res2 = minimize(
                    lambda x: _compute_bound(x[0], x[1], x[2]),
                    res.x,
                    method='L-BFGS-B',
                    options={'gtol': 1e-12, 'maxiter': 1000}
                )
                update(res2.x, res2.fun)
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                lambda x: _compute_bound(x[0], x[1], x[2]),
                best,
                method='Nelder-Mead',
                options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 10000, 'adaptive': True}
            )
            update(res_final.x, res_final.fun)
        except Exception:
            pass
    
    if not np.all(np.isfinite(best)):
        best = seed.copy()
    
    return best
