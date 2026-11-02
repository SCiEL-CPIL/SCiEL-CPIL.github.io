"""Optimized program: uses exact Gaussian quadrature/moments to compute the
bound analytically, and Differential Evolution for global optimization.
This avoids numerical FFT errors and local minima traps."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution
import numpy.polynomial.hermite as hermite

def _get_monomial_coeffs(herm_coeffs):
    """
    Given coefficients for H0, H4, H8, H12, return the monomial coefficients
    of the polynomial P(x) = sum c_i H_{n_i}(x).
    """
    # herm_coeffs is [a0, a1, a2, a3] for degrees 0, 4, 8, 12
    # We construct the full Hermite series coefficients vector up to degree 12
    c_full = np.zeros(13)
    c_full[0] = herm_coeffs[0]
    c_full[4] = herm_coeffs[1]
    c_full[8] = herm_coeffs[2]
    c_full[12] = herm_coeffs[3]
    
    # Convert to monomial basis
    # herm2poly expects coefficients in increasing order of degree
    poly_coeffs = hermite.herm2poly(c_full)
    return poly_coeffs

def _poly_sq_coeffs(poly_coeffs):
    """Return monomial coefficients of P(x)^2."""
    return np.convolve(poly_coeffs, poly_coeffs)

def _poly_4_coeffs(poly_coeffs):
    """Return monomial coefficients of P(x)^4."""
    p2 = _poly_sq_coeffs(poly_coeffs)
    return np.convolve(p2, p2)

def _gaussian_moment(alpha, k):
    """
    Compute integral of x^k * exp(-alpha * x^2) from -inf to inf.
    Returns 0 if k is odd.
    """
    if k % 2 != 0:
        return 0.0
    m = k // 2
    # Formula: (2m)! / (2^(2m) m! alpha^m) * sqrt(pi/alpha)
    # Using math.factorial or scipy.special.factorial for stability?
    # For small m (up to 24), factorial is fine.
    import math
    num = math.factorial(2*m)
    den = (2**(2*m)) * math.factorial(m) * (alpha**m)
    val = (num / den) * np.sqrt(np.pi / alpha)
    return val

def _integrate_gaussian_poly(alpha, poly_coeffs):
    """
    Compute integral of P(x) * exp(-alpha * x^2) dx.
    poly_coeffs are monomial coefficients c_k for x^k.
    """
    integral = 0.0
    for k, c in enumerate(poly_coeffs):
        if c != 0.0:
            integral += c * _gaussian_moment(alpha, k)
    return integral

def _compute_bound(coeffs):
    """
    Compute the upper bound C4 for the given Hermite coefficients.
    coeffs: [a0, a1, a2] for H0, H4, H8.
    a3 is derived to satisfy P(0)=0.
    """
    a0, a1, a2 = coeffs
    
    # H_n(0) for Physicists' Hermite
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 665280.0
    
    # a0*H0(0) + a1*H4(0) + a2*H8(0) + a3*H12(0) = 0
    a3 = -(a0 * H0_0 + a1 * H4_0 + a2 * H8_0) / H12_0
    
    full_coeffs = np.array([a0, a1, a2, a3])
    
    # Get monomial coefficients of P(x)
    p_coeffs = _get_monomial_coeffs(full_coeffs)
    
    # Calculate A = int e^{-x^2} P(x)^2 dx
    p2_coeffs = _poly_sq_coeffs(p_coeffs)
    A = _integrate_gaussian_poly(1.0, p2_coeffs)
    
    # Calculate B = int e^{-2x^2} P(x)^4 dx
    p4_coeffs = _poly_4_coeffs(p_coeffs)
    B = _integrate_gaussian_poly(2.0, p4_coeffs)
    
    if A <= 0:
        return 1e18
        
    # Bound = 2 * pi * B / A
    return 2.0 * np.pi * B / A

def construct():
    # Define bounds for the 3 free coefficients
    bounds = [(-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0)]
    
    # Use Differential Evolution for global search
    # maxiter controls the number of generations
    # popsize controls the population size
    res = differential_evolution(
        _compute_bound,
        bounds,
        seed=42,
        maxiter=100,
        popsize=15,
        tol=1e-10,
        mutation=(0.5, 1.0),
        recombination=0.7
    )
    
    best_coeffs = res.x
    
    # Optional: Local refinement with Nelder-Mead or Powell
    from scipy.optimize import minimize
    try:
        res_local = minimize(
            _compute_bound,
            best_coeffs,
            method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12}
        )
        if res_local.fun < res.fun:
            best_coeffs = res_local.x
    except Exception:
        pass
        
    return np.asarray(best_coeffs, dtype=float)
# EVOLVE-BLOCK-END
