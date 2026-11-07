"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # params length 10: use a rich polynomial basis with all coefficients active.
    # x is 1D array of inputs. We compute powers x^0 .. x^5 and their absolute variants
    # to capture non-polynomial or asymmetric behavior, but keep it smooth and finite.
    # Basis:
    #   p0 = 1
    #   p1 = x
    #   p2 = x^2
    #   p3 = x^3
    #   p4 = x^4
    #   p5 = x^5
    #   p6 = |x|
    #   p7 = |x|^3  (smooth-ish, helps with asymmetric noise-free targets)
    #   p8 = sin(x)
    #   p9 = cos(x)
    
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.array([], dtype=np.float64)

    # Use Chebyshev polynomials of the first kind for better conditioning
    # T_0(x) = 1
    # T_1(x) = x
    # T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
    
    # Map x from [-3, 3] to [-1, 1] for Chebyshev
    t = x / 3.0
    
    # Compute Chebyshev polynomials T_0 through T_9
    # We store them in a matrix where rows are points and columns are polynomial degrees
    basis = np.zeros((len(x), 10), dtype=np.float64)
    basis[:, 0] = 1.0
    if len(x) > 0:
        basis[:, 1] = t
        for n in range(2, 10):
            basis[:, n] = 2.0 * t * basis[:, n-1] - basis[:, n-2]
            
    # Orthonormalize the basis using QR decomposition.
    # This transforms the parameter space such that the Hessian of the MSE
    # with respect to the parameters is proportional to the identity matrix
    # (assuming full rank), which greatly improves the convergence of BFGS.
    # The span of the functions remains identical to the original Chebyshev basis.
    try:
        Q, R = np.linalg.qr(basis)
        # Q has shape (N, 10) with orthonormal columns
        pred = Q @ params
    except np.linalg.LinAlgError:
        # Fallback to raw basis if QR fails
        pred = basis @ params
        
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        pred = np.zeros_like(x, dtype=np.float64)
        
    return pred
# EVOLVE-BLOCK-END
