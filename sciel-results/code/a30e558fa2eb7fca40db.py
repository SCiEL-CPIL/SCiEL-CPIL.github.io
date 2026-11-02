"""Rewritten program using Basin Hopping for global search on the uncertainty bound."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize, basinhopping

def _get_hermite_norm_sq(n):
    """Returns 2^n * n! (excluding sqrt(pi) which cancels in the ratio)."""
    return (2.0 ** n) * math.factorial(n)

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product Delta_x * Delta_xi for the test function.
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    Constraint: f(0) = 0 => P(0) = 0.
    H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128 (Physicists' Hermite).
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (i)^n * H_n(xi)*exp(-xi^2/2).
    For even n in {0, 4, 8, 12}, (i)^n = 1.
    Thus the function is self-Fourier (up to phase 1).
    Delta_x^2 = Delta_xi^2 = <x^2> = integral x^2 f^2 dx / ||f||_2^2.
    Product = <x^2>.
    """
    # Determine a12 from P(0) = 0
    # a0*1 + a4*12 + a8*1680 + a12*2032128 = 0
    a12 = -(a0 + 12.0 * a4 + 1680.0 * a8) / 2032128.0
    
    coeffs = np.array([a0, a4, a8, a12])
    degrees = np.array([0, 4, 8, 12])
    
    # 1. Calculate ||f||_2^2
    norms_sq = np.array([_get_hermite_norm_sq(d) for d in degrees])
    norm_total = np.sum(coeffs**2 * norms_sq)
    
    if norm_total < 1e-30:
        return 1e10
        
    # 2. Calculate <x^2> = integral x^2 f^2 dx / ||f||_2^2
    # Orthogonality of Hermite polynomials: <H_n H_m> = 0 if n != m.
    # x^2 H_n(x) = 1/4 H_{n+2}(x) + n H_n(x) + n(n-1) H_{n-2}(x)
    # We need <x^2 f, f>. Since f is a sum of H_0, H_4, H_8, H_12,
    # x^2 f will contain terms H_{n+2}, H_n, H_{n-2}.
    # Only terms with the same degree as components of f contribute to the inner product.
    
    # Contribution from a_n H_n:
    # The H_n part of x^2 (a_n H_n) is a_n * n * H_n.
    # This overlaps with a_n H_n in f.
    # Integral is a_n * (a_n * n) * ||H_n||^2.
    
    # Are there cross terms?
    # x^2 (a_0 H_0) = 1/4 H_2. No H_0, H_4... in f. No contribution.
    # x^2 (a_4 H_4) = 1/4 H_6 + 4 H_4 + 12 H_2. Overlaps with a_4 H_4.
    # x^2 (a_8 H_8) = 1/4 H_10 + 8 H_8 + 56 H_6. Overlaps with a_8 H_8.
    # x^2 (a_12 H_12) = 1/4 H_14 + 12 H_12 + 132 H_10. Overlaps with a_12 H_12.
    
    # So, <x^2 f, f> = 4 a_4^2 ||H_4||^2 + 8 a_8^2 ||H_8||^2 + 12 a_12^2 ||H_12||^2
    
    norm_H4_sq = _get_hermite_norm_sq(4)
    norm_H8_sq = _get_hermite_norm_sq(8)
    norm_H12_sq = _get_hermite_norm_sq(12)
    
    integral_x2_f2 = (4.0 * a4**2 * norm_H4_sq + 
                      8.0 * a8**2 * norm_H8_sq + 
                      12.0 * a12**2 * norm_H12_sq)
    
    variance = integral_x2_f2 / norm_total
    
    return variance

def _objective(x):
    """Objective function for optimizer."""
    a0, a4, a8 = x
    return _compute_bound(a0, a4, a8)

def construct():
    """
    Uses Basin Hopping to find the global minimum of the uncertainty bound.
    Starts from the known good seed point and explores the landscape.
    """
    start_time = time.time()
    deadline = start_time + 180.0  # 30s safety margin for 210s limit

    # Seed values from "Prior state of the art"
    # These are known good starting points
    x0 = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _compute_bound(*x0)
    best_x = x0.copy()

    # Define a step size for basinhopping
    # The coefficients vary in scale, so we use a relative step or fixed small step
    # a0 ~ 1, a4 ~ 0.1, a8 ~ 0.001
    # Let's use a step generator that respects scales
    def step(x):
        # Random perturbation
        delta = np.random.normal(0, 0.1, size=3)
        # Scale the perturbations
        delta[1] *= 0.1
        delta[2] *= 0.01
        return x + delta

    try:
        # Run basinhopping
        # niter=100 is a reasonable default, but we check time
        res = basinhopping(_objective, x0, niter=100, stepfun=step, 
                           minimizer_kwargs={'method': 'BFGS', 'options': {'gtol': 1e-8}})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            
    except Exception:
        pass

    # Check time and if we have time for more local refinement or restarts
    # If basinhopping didn't improve, try a few random restarts
    if time.time() < deadline - 30.0:
        for i in range(20):
            if time.time() > deadline - 10.0:
                break
            # Random restart near the best found or seed
            if i % 2 == 0:
                x_start = best_x + np.random.normal(0, 0.05, size=3)
            else:
                x_start = x0 + np.random.normal(0, 0.1, size=3)
            
            try:
                res_local = minimize(_objective, x_start, method='BFGS')
                if res_local.fun < best_val:
                    best_val = res_local.fun
                    best_x = res_local.x
            except Exception:
                continue

    return best_x
# EVOLVE-BLOCK-END
