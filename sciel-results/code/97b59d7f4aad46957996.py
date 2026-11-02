"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

# Pre-compute factorial and power of 2 for Hermite norms
# Norm of H_n is sqrt(2^n * n! * sqrt(pi))
# We work with the squared norm: 2^n * n! * sqrt(pi)
# For the ratio, sqrt(pi) cancels out.

def _get_hermite_norm_sq(n):
    """Returns 2^n * n! (excluding sqrt(pi) which cancels in the ratio)."""
    return (2.0 ** n) * math.factorial(n)

def _compute_bound(a0, a4, a8):
    """
    Compute the uncertainty product Delta_x * Delta_xi for the test function.
    f(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    Constraint: f(0) = 0 => P(0) = 0.
    H0(0)=1, H4(0)=12, H8(0)=1680, H12(0)=2032128 (Physicists' Hermite).
    
    The Fourier transform of H_n(x)*exp(-x^2/2) is (i)^n * H_n(xi)*exp(-xi^2/2)
    (up to a constant factor that cancels in the ratio).
    For even n, (i)^n is real: (i)^0=1, (i)^4=1, (i)^8=1, (i)^12=1.
    So the Fourier-space coefficients are the same as position-space.
    Thus Delta_x^2 = Delta_xi^2, and the product is <x^2>.
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
    # x^2 H_n = 1/4 H_{n+2} + n H_n + n(n-1) H_{n-2}
    # Overlaps with f (degrees 0, 4, 8, 12):
    # H_4: from x^2 H_4 -> 4*H_4
    # H_8: from x^2 H_8 -> 8*H_8
    # H_12: from x^2 H_12 -> 12*H_12
    
    coeff_x2f_H4 = 4.0 * a4
    coeff_x2f_H8 = 8.0 * a8
    coeff_x2f_H12 = 12.0 * a12
    
    norm_H4_sq = _get_hermite_norm_sq(4)
    norm_H8_sq = _get_hermite_norm_sq(8)
    norm_H12_sq = _get_hermite_norm_sq(12)
    
    integral_x2_f2 = (coeff_x2f_H4 * a4 * norm_H4_sq + 
                      coeff_x2f_H8 * a8 * norm_H8_sq + 
                      coeff_x2f_H12 * a12 * norm_H12_sq)
    
    variance = integral_x2_f2 / norm_total

    return variance

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # Safety margin before 180s limit
    
    # Seed values from "Prior state of the art"
    # These are known good starting points
    seed_coeffs = np.array([-1.13, 0.04, 1.0/3240.0])
    best_val = _compute_bound(*seed_coeffs)
    best_coeffs = seed_coeffs.copy()
    
    # Helper to check time
    def time_exceeded():
        return time.time() > deadline

    # 1. Initial Local Refinement from Seed
    try:
        res = minimize(
            _compute_bound,
            seed_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-15}
        )
        if res.fun < best_val:
            best_val = res.fun
            best_coeffs = res.x
    except Exception:
        pass

    if time_exceeded():
        return np.array(best_coeffs)

    # 2. Multi-Start CMA-ES Global Search
    # CMA-ES is robust for non-convex, low-dimensional optimization.
    # We define bounds to keep the search within reasonable physical limits.
    # The coefficients are expected to be small.
    bounds_lower = np.array([-5.0, -5.0, -5.0])
    bounds_upper = np.array([5.0, 5.0, 5.0])
    
    # We will run multiple CMA-ES instances with different random seeds and initial means
    rng = np.random.default_rng(42)
    
    # Define a wrapper for CMA-ES that respects bounds by penalizing out-of-bounds
    # or simply clipping, but CMA-ES handles bounds natively in newer versions.
    # However, to be safe and compatible, we use the bounds parameter if available,
    # or rely on the fact that the function is smooth and we can guide it.
    # scipy.optimize doesn't have CMA-ES built-in in standard versions, 
    # but 'cma' library might not be installed. 
    # Wait, the prompt says "Python standard library, NumPy and SciPy are available."
    # It does NOT explicitly say 'cma' package is available.
    # I must implement CMA-ES logic or use a robust alternative from SciPy.
    # SciPy has 'differential_evolution', 'shgo', 'dual_annealing', 'basinhopping'.
    # The directive specifically suggested CMA-ES. If 'cma' is not available, 
    # I should use a robust global optimizer from SciPy that mimics its strengths,
    # or implement a simple CMA-ES. 
    # Given the constraints, implementing a full CMA-ES from scratch is risky for bugs.
    # Let's check if `cma` is a common dependency. Usually not in standard SciPy eval environments.
    # I will use `scipy.optimize.dual_annealing` or `differential_evolution` with careful tuning,
    # OR I can implement a simplified version of CMA-ES.
    # Actually, `scipy.optimize` has `basinhopping` which is good for escaping local minima.
    # Let's try to use `differential_evolution` with a larger population and tighter tolerances,
    # combined with `basinhopping` for local polishing.
    
    # Let's try `differential_evolution` first with better parameters.
    try:
        from scipy.optimize import differential_evolution
        de_result = differential_evolution(
            _compute_bound,
            bounds=[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
            seed=42,
            maxiter=200,
            tol=1e-12,
            atol=1e-12,
            popsize=30,
            mutation=(0.5, 1.5),
            recombination=0.7,
            polish=True,
            init='latinhypercube',
            updating='deferred'
        )
        if de_result.fun < best_val:
            best_val = de_result.fun
            best_coeffs = de_result.x
    except Exception:
        pass

    if time_exceeded():
        return np.array(best_coeffs)

    # 3. Basin Hopping for further refinement
    # Basin hopping is effective for finding global minima in complex landscapes
    try:
        from scipy.optimize import basinhopping
        # Use the best point found so far as the starting point
        res_bh = basinhopping(
            _compute_bound,
            x0=best_coeffs,
            niter=50,
            T=1.0,
            stepsize=0.5,
            minimizer_kwargs={'method': 'L-BFGS-B', 'options': {'ftol': 1e-15, 'gtol': 1e-15}},
            seed=42
        )
        if res_bh.fun < best_val:
            best_val = res_bh.fun
            best_coeffs = res_bh.x
    except Exception:
        pass

    if time_exceeded():
        return np.array(best_coeffs)

    # 4. Final High-Precision Polish
    # Use Powell's method for a derivative-free, high-precision local search
    try:
        res_pw = minimize(
            _compute_bound,
            best_coeffs,
            method='Powell',
            options={'maxiter': 10000, 'xtol': 1e-15, 'ftol': 1e-15}
        )
        if res_pw.fun < best_val:
            best_val = res_pw.fun
            best_coeffs = res_pw.x
    except Exception:
        pass

    # 5. One last check with L-BFGS-B for very tight convergence
    try:
        res_final = minimize(
            _compute_bound,
            best_coeffs,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-18}
        )
        if res_final.fun < best_val:
            best_val = res_final.fun
            best_coeffs = res_final.x
    except Exception:
        pass

    return np.array(best_coeffs)

# EVOLVE-BLOCK-END
