"""Symbolic regression with 1 input; Scaled Chebyshev polynomial basis for improved BFGS convergence."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model using scaled Chebyshev polynomial basis (degrees 0-9) for well-conditioned fitting.
    
    y = sum_{k=0}^{9} p_k * s_k * T_k(2x/L - 1)
    
    where s_k are fixed scaling factors that normalize each basis function's magnitude,
    improving BFGS convergence from the all-ones starting point.
    
    The scaling ensures each basis function has comparable magnitude over the domain,
    making the parameter scales more uniform and helping gradient-based optimization.
    """
    x = np.asarray(x, dtype=float)
    
    # Map x from [0.1, 6] to [-1, 1] for Chebyshev evaluation
    L = 6.0
    z = 2.0 * x / L - 1.0
    z = np.clip(z, -1.0, 1.0)
    
    # Compute Chebyshev polynomials T_0 through T_9 using the recurrence relation
    T = np.zeros((10, x.shape[0]))
    T[0, :] = 1.0
    T[1, :] = z
    for n in range(2, 10):
        T[n, :] = 2.0 * z * T[n-1, :] - T[n-2, :]
    
    # Scale each basis function by the inverse of its typical magnitude over the domain.
    # T_k(z) for z in [-1,1] has max |T_k| = 1, but the RMS value varies.
    # Use a fixed scaling based on the expected RMS over the training domain.
    # For z = 2x/6 - 1 with x in [0.1, 6], z is in [-0.967, 1.0].
    # Approximate RMS of T_k over this range; use simple heuristics:
    # T_0: RMS ~ 1.0
    # T_1: RMS ~ 0.5 (z ranges roughly -1 to 1, RMS of z ~ 0.5)
    # Higher degrees: RMS decreases roughly as 1/sqrt(k) for uniform z
    # Use conservative scaling to avoid over-amplifying noise
    scales = np.array([1.0, 1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15])
    
    # Apply scaling to basis functions
    T_scaled = T * scales[:, np.newaxis]
    
    # Linear combination with the 10 parameters
    y = np.dot(params, T_scaled)
    
    return y
# EVOLVE-BLOCK-END
