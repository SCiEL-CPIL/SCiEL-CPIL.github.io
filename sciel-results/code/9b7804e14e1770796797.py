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
    
    # params is length 10; clip to avoid overflow during optimization
    # Chebyshev coefficients can be larger than monomial coefficients for the same function,
    # but typically remain bounded for smooth functions on [-1, 1].
    # Clipping at 1000 is safer than 100 to allow for more expressive fits if needed,
    # but 100 is likely sufficient and prevents extreme values.
    params = np.clip(np.asarray(params, dtype=np.float64), -100.0, 100.0)
    
    pred = basis @ params
    
    # Ensure finite output
    if not np.all(np.isfinite(pred)):
        # Fallback: use only the first two terms (linear in t, so linear in x)
        # pred = params[0] * 1 + params[1] * t
        pred = np.clip(params[0], -1e6, 1e6) + np.clip(params[1], -1e6, 1e6) * t
    
    return pred
# EVOLVE-BLOCK-END
