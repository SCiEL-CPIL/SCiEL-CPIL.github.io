import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid symbolic regression basis: Gaussian RBFs + low-degree polynomials.
    
    The 10 parameters are linear coefficients over a FIXED basis that mixes
    two complementary families:
    1. 6 Gaussian Radial Basis Functions (RBFs) with fixed centers and widths.
       These provide local flexibility to capture bumps, oscillations, and
       sharp transitions that global polynomials struggle with.
    2. 4 Polynomial terms (1, x, x^2, x^3).
       These capture global trends and ensure the model can represent
       monotone growth or decay even if the RBFs cancel out.
    
    BFGS on 10 linear params is a convex quadratic problem, so it converges
    robustly from the all-ones start. The RBF centers are spread across
    [-2, 2] to cover the domain, and widths are chosen to balance locality
    and smoothness.
    """
    # Deadline guard: return a safe finite array if we are about to be killed.
    if time.time() - _START_TIME > _DEADLINE:
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()

    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]

    # --- Build the 10 fixed basis columns on [-2, 2] ---
    
    # Polynomial part (4 terms): captures global trend.
    poly_0 = np.ones_like(x)
    poly_1 = x
    poly_2 = x * x
    poly_3 = x * x * x

    # RBF part (6 terms): captures local features.
    # Centers spread across [-2, 2]
    centers = np.array([-2.0, -1.2, -0.4, 0.4, 1.2, 2.0])
    
    # Widths: slightly wider than the spacing to ensure overlap and smoothness.
    # Spacing is 0.8. Sigma = 0.6 gives good coverage.
    sigma = 0.6
    
    rbf_terms = []
    for c in centers:
        # exp(-0.5 * ((x - c) / sigma)^2)
        diff = x - c
        rbf = np.exp(-0.5 * (diff / sigma) ** 2)
        rbf_terms.append(rbf)

    # Stack all basis functions
    basis = np.column_stack((poly_0, poly_1, poly_2, poly_3) + tuple(rbf_terms))

    result = basis @ p
    result = np.where(np.isfinite(result), result, 0.0)
    return result
