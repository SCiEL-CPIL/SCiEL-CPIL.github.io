"""Optimized program: analytically computes integrals for the Fourier-invariant
Hermite-Gaussian test function. Uses scipy.optimize to minimize the bound
C = sqrt(int|g|^4) * sqrt(int|g_hat|^4) / int|g|^2.
The degree-12 coefficient is fixed so that g(0) = 0."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import eval_hermite
from scipy.integrate import quad


def _hermite_combo(a0, a1, a2):
    """Return coefficients for H0, H4, H8, H12 with g(0)=0 (physicists' H)."""
    # H_n(0) for even n (physicists): H0=1, H4=-12, H8=1680, H12=-2075040
    H0_0, H4_0, H8_0, H12_0 = 1.0, -12.0, 1680.0, -2075040.0
    # a0*H0(0)+a1*H4(0)+a2*H8(0)+a3*H12(0)=0  =>  a3 = -(a0*H0_0 + a1*H4_0 + a2*H8_0)/H12_0
    a3 = -(a0 * H0_0 + a1 * H4_0 + a2 * H8_0) / H12_0
    return np.array([a0, a1, a2, a3])


def _poly_val(x, coeffs):
    """Evaluate the polynomial part P(x) = sum a_i H_i(x)."""
    a0, a1, a2, a3 = coeffs
    return (a0 * eval_hermite(0, x) + 
            a1 * eval_hermite(4, x) + 
            a2 * eval_hermite(8, x) + 
            a3 * eval_hermite(12, x))


def _bound(coeffs):
    """
    Compute the uncertainty bound using analytical Fourier properties.
    g(x) = exp(-x^2/2) * P(x)
    g_hat(k) = sqrt(2*pi) * exp(-k^2/2) * Q(k)
    where Q(k) = sum a_i (-i)^i H_i(k)
    """
    coeffs = np.asarray(coeffs, dtype=float)
    if np.any(np.isnan(coeffs)) or np.any(np.isinf(coeffs)):
        return 1e18
        
    a0, a1, a2, a3 = coeffs
    
    # Integrand for int |g|^4 dx
    # |g(x)|^4 = exp(-2x^2) * P(x)^4
    def integrand_g4(x):
        p = _poly_val(x, coeffs)
        return np.exp(-2.0 * x * x) * (p ** 4)
        
    # Integrand for int |g|^2 dx
    # |g(x)|^2 = exp(-x^2) * P(x)^2
    def integrand_g2(x):
        p = _poly_val(x, coeffs)
        return np.exp(-x * x) * (p ** 2)
        
    # Integrand for int |g_hat|^4 dk
    # |g_hat(k)|^4 = (2*pi)^2 * exp(-2k^2) * |Q(k)|^4
    # Q(k) is complex. We compute |Q(k)|^2 = Q(k) * conj(Q(k))
    # Since H_i(k) are real for real k, Q(k) = sum a_i (-i)^i H_i(k)
    # Let's compute the real and imaginary parts or just use complex arithmetic.
    def integrand_G4(k):
        # (-i)^0 = 1, (-i)^4 = 1, (-i)^8 = 1, (-i)^12 = 1
        # All degrees are multiples of 4, so (-i)^n = 1 for all n in {0,4,8,12}
        # Thus Q(k) is real and equal to P(k)!
        # Let's verify: (-i)^4 = (-i)^2 * (-i)^2 = -1 * -1 = 1. Correct.
        # So |Q(k)|^4 = P(k)^4.
        p = _poly_val(k, coeffs)
        return (2.0 * np.pi) ** 2 * np.exp(-2.0 * k * k) * (p ** 4)
        
    # Integrate over a sufficiently large range
    # The Gaussian decays fast. 10 sigma is enough for high precision.
    limit = 10.0
    
    try:
        int_g4, _ = quad(integrand_g4, -limit, limit, limit=200, epsabs=1e-12, epsrel=1e-12)
        int_g2, _ = quad(integrand_g2, -limit, limit, limit=200, epsabs=1e-12, epsrel=1e-12)
        int_G4, _ = quad(integrand_G4, -limit, limit, limit=200, epsabs=1e-12, epsrel=1e-12)
    except Exception:
        return 1e18
        
    if int_g2 <= 1e-15:
        return 1e18
        
    return np.sqrt(int_g4) * np.sqrt(int_G4) / int_g2


def construct():
    # Multistart optimization of the three free coefficients.
    best = None
    best_val = np.inf
    rng = np.random.default_rng(42)
    
    # Literature seed and perturbations
    lit_seed = np.array([-113 / 100.0, 1 / 25.0, 1 / 3240.0])
    
    starts = [
        lit_seed,
        lit_seed * 0.9,
        lit_seed * 1.1,
        lit_seed + rng.normal(scale=0.1, size=3),
        lit_seed + rng.normal(scale=0.1, size=3),
        lit_seed + rng.normal(scale=0.1, size=3),
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
    ]
    
    for s in starts:
        s = np.asarray(s, dtype=float)
        try:
            # Use L-BFGS-B for robustness and speed with gradient estimation
            # Or Nelder-Mead which doesn't require gradients.
            # Given the smoothness, Nelder-Mead is often sufficient and robust.
            res = minimize(
                _bound,
                s,
                method="Nelder-Mead",
                options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True},
            )
            val = _bound(res.x)
            if val < best_val:
                best_val = val
                best = res.x
        except Exception:
            continue
            
    if best is None:
        return np.array([-113 / 100.0, 1 / 25.0, 1 / 3240.0])
        
    # Final refinement with Powell
    try:
        res2 = minimize(
            _bound,
            best,
            method="Powell",
            options={"maxiter": 2000, "ftol": 1e-14, "xtol": 1e-14},
        )
        val2 = _bound(res2.x)
        if val2 < best_val:
            best = res2.x
            best_val = val2
    except Exception:
        pass
        
    return np.asarray(best, dtype=float)
# EVOLVE-BLOCK-END
