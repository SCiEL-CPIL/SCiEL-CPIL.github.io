"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Normalized Chebyshev polynomial basis on [-2, 2] mapped to [-1, 1].
    
    Each basis function is scaled to have unit RMS over the domain, so all
    parameters control components of similar magnitude. This improves the
    conditioning of the Hessian for BFGS, allowing more reliable convergence.
    
    We use 10 normalized Chebyshev polynomials T_0 through T_9.
    """
    t = x / 2.0  # Map from [-2, 2] to [-1, 1]
    n = x.shape[0]
    
    # Compute Chebyshev polynomials T_k(t) for k=0..9
    # T_0(t) = 1
    # T_1(t) = t
    # T_{k+1}(t) = 2*t*T_k(t) - T_{k-1}(t)
    
    T0 = np.ones_like(t)
    T1 = t
    
    # Build array of Chebyshev polynomials
    cheb = [T0, T1]
    for k in range(2, 10):
        Tk = 2.0 * t * cheb[-1] - cheb[-2]
        cheb.append(Tk)
    
    # Stack basis functions: shape (n_points, 10)
    basis = np.stack(cheb, axis=-1)
    
    # Normalize each basis function to have unit RMS over the domain
    # This ensures parameters are at similar scales, improving BFGS conditioning
    rms = np.sqrt(np.mean(basis ** 2, axis=0))  # shape (10,)
    rms = np.where(rms < 1e-12, 1.0, rms)  # avoid division by zero
    basis_normalized = basis / rms
    
    # Linear combination
    result = basis_normalized @ params
    
    return result
# EVOLVE-BLOCK-END
