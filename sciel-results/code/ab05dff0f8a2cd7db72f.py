"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: 5 Chebyshev polynomials (mapped to [-1,1]) + 5 RBF terms.
    
    Chebyshev polynomials T_0..T_4 provide a well-conditioned polynomial basis
    that captures global smooth trends. The RBF terms with fixed centers provide
    local flexibility to capture non-polynomial features.
    
    Basis:
    - T_0(t) = 1, T_1(t) = t, T_2(t) = 2t^2-1, T_3(t) = 4t^3-3t, T_4(t) = 8t^4-8t^2+1
      where t = x/2 (maps [-2,2] to [-1,1])
    - RBF: exp(-sigma^2 * (x - c_i)^2) for c_i in {-1.5, -0.75, 0, 0.75, 1.5}
    
    This is linear in params, so BFGS should converge reliably.
    The Chebyshev basis is orthogonal with respect to the weight 1/sqrt(1-t^2),
    and well-conditioned. The RBF terms add local adaptability.
    """
    n = len(x)
    
    # Map x from [-2, 2] to [-1, 1]
    t = x / 2.0
    
    # Build Chebyshev basis T_0..T_4
    # T_0 = 1
    # T_1 = t
    # T_2 = 2*t^2 - 1
    # T_3 = 4*t^3 - 3*t
    # T_4 = 8*t^4 - 8*t^2 + 1
    basis = np.empty((n, 10), dtype=np.float64)
    
    t2 = t * t
    t3 = t2 * t
    t4 = t2 * t2
    
    basis[:, 0] = 1.0
    basis[:, 1] = t
    basis[:, 2] = 2.0 * t2 - 1.0
    basis[:, 3] = 4.0 * t3 - 3.0 * t
    basis[:, 4] = 8.0 * t4 - 8.0 * t2 + 1.0
    
    # RBF terms with fixed centers
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    sigma = 1.0  # width parameter
    
    for i, c in enumerate(centers):
        basis[:, 5 + i] = np.exp(-sigma**2 * (x - c)**2)
    
    # Linear combination
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
