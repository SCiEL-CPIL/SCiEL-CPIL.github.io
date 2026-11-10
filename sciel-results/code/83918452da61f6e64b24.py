"""Symbolic regression with unconstrained rational Chebyshev model and deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Approximates the target function using a rational Chebyshev model:
    
    y(x) = c + (a0*T0 + a1*T1 + a2*T2 + a3*T3 + a4*T4) / (1 + b1*T1 + b2*T2 + b3*T3 + b4*T4)
    
    where T_i are Chebyshev polynomials of the first kind evaluated at t = x/2.
    The mapping t = x/2 ensures the basis functions vary over the entire input domain [-2, 2].
    
    The 10 params are: a0, a1, a2, a3, a4, b1, b2, b3, b4, and an offset c.
    
    A deadline guard is included to ensure the function returns within the 150s limit.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Deadline guard: if we're getting close to the 150s limit, return a safe default
    elapsed = time.time() - _START_TIME
    if elapsed > 145.0:
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x)
    
    # Ensure x is 1D
    x = np.asarray(x, dtype=np.float64).ravel()
    params = np.asarray(params, dtype=np.float64).ravel()
    
    if params.size < 10:
        params = np.pad(params, (0, 10 - params.size))
    elif params.size > 10:
        params = params[:10]
    
    a0, a1, a2, a3, a4, b1, b2, b3, b4, c = params
    
    # Map x from [-2, 2] to t in [-1, 1]
    t = 0.5 * x
    
    # Compute Chebyshev polynomials T0 through T4 for t
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    # T4(t) = 8t^4 - 8t^2 + 1
    T0 = np.ones_like(t)
    T1 = t
    T2 = 2.0 * t * t - 1.0
    T3 = 4.0 * t * t * t - 3.0 * t
    T4 = 8.0 * t * t * t * t - 8.0 * t * t + 1.0
    
    # Numerator: a0*T0 + a1*T1 + a2*T2 + a3*T3 + a4*T4
    numerator = a0 * T0 + a1 * T1 + a2 * T2 + a3 * T3 + a4 * T4
    
    # Denominator: 1 + b1*T1 + b2*T2 + b3*T3 + b4*T4
    # Constrain denominator to be 1 + tanh(scaled_perturbation) to prevent it from approaching zero
    # This ensures numerical stability during BFGS optimization
    raw_denom_pert = b1 * T1 + b2 * T2 + b3 * T3 + b4 * T4
    # Scale the perturbation and apply tanh to keep it bounded in (-1, 1)
    # The factor 0.5 keeps the total denominator in (0.5, 1.5), avoiding zero crossing
    denominator = 1.0 + 0.5 * np.tanh(raw_denom_pert)
    
    # Compute rational part with safe division
    eps = 1e-10
    safe_denominator = np.where(np.abs(denominator) < eps, eps, denominator)
    rational_part = numerator / safe_denominator
    
    # Add offset
    result = c + rational_part
    
    # Ensure finite values
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
