"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    import numpy as np
    from scipy.special import roots_genlaguerre
    from scipy.optimize import minimize
    import time

    start_time = time.time()
    deadline = start_time + 570

    m = 12
    n = 1  # alpha parameter for generalized Laguerre

    # Get the roots of L_m^(n)(x) = L_12^1(x)
    # These are the natural double-root locations when squared
    try:
        r, w = roots_genlaguerre(m, n)
        r = np.sort(r)  # ensure ascending order
        # All roots should be positive for Laguerre polynomials
        if np.all(r > 0):
            return list(r)
    except Exception:
        pass

    # Fallback: compute roots numerically via polynomial
    def laguerre_coeffs(m, alpha):
        """Return coefficients of L_m^(alpha)(x) in descending powers of x."""
        if m == 0:
            return np.array([1.0])
        if m == 1:
            return np.array([1.0, -(alpha + 1.0)])
        # Recurrence: m*L_m = (2m-1+alpha-x)*L_{m-1} - (m-1+alpha)*L_{m-2}
        # So: L_m = ((2m-1+alpha-x)*L_{m-1} - (m-1+alpha)*L_{m-2}) / m
        c_prev2 = np.array([1.0])  # L_0
        c_prev1 = np.array([1.0, -(alpha + 1.0)])  # L_1
        for k in range(2, m + 1):
            # (2k-1+alpha-x) * L_{k-1}
            # L_{k-1} has degree k-1, so (2k-1+alpha-x)*L_{k-1} has degree k
            # Coefficients: -1 * c_prev1 shifted, plus (2k-1+alpha) * c_prev1
            term1 = np.zeros(k + 1)
            term1[:-1] = -c_prev1  # -x * L_{k-1}
            term1[1:] += (2*k - 1 + alpha) * c_prev1  # (2k-1+alpha) * L_{k-1}
            # Actually let me redo this more carefully
            # L_{k-1}(x) = sum_{j=0}^{k-1} a_j x^j (a_{k-1} != 0)
            # (2k-1+alpha - x) * L_{k-1}(x) = (2k-1+alpha)*L_{k-1}(x) - x*L_{k-1}(x)
            # If c_prev1 = [a_{k-1}, a_{k-2}, ..., a_0] (descending powers)
            # Then (2k-1+alpha)*c_prev1 gives coefficients for degrees k-1 down to 0
            # x*c_prev1 gives coefficients for degrees k down to 1 (shifted)
            
            c_curr = np.zeros(k + 1)
            # (2k-1+alpha) * L_{k-1}: contributes to degrees 0..k-1
            c_curr[1:] += (2*k - 1 + alpha) * c_prev1
            # -x * L_{k-1}: contributes to degrees 1..k
            c_curr[:-1] += -c_prev1
            # Subtract (k-1+alpha) * L_{k-2}
            c_curr[2:] -= (k - 1 + alpha) * c_prev2
            # Divide by k
            c_curr /= k
            c_prev2, c_prev1 = c_prev1, c_curr
        return c_prev1

    try:
        coeffs = laguerre_coeffs(m, n)
        roots = np.roots(coeffs)
        # Filter real positive roots
        real_roots = np.sort(roots[np.abs(roots.imag) < 1e-10].real)
        real_roots = real_roots[real_roots > 0]
        if len(real_roots) == m:
            return list(real_roots)
    except Exception:
        pass

    # Final fallback: arithmetic progression
    return [4.0 * (i + 1) for i in range(12)]

# EVOLVE-BLOCK-END
