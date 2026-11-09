"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 150.0
_TIME_MARGIN = 5.0  # safety margin in seconds


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Shifted Legendre polynomial basis with normalized inputs.
    
    Structural change: Use shifted Legendre polynomials (degree 0-3) on each
    normalized input, plus one key cross-interaction term. This provides an
    orthogonal, well-conditioned basis that captures cubic-level curvature
    and interactions more efficiently than raw monomials.
    
    Normalization: Map inputs from [0.5, 2] to [0, 1] via (x - 0.5) / 1.5
    This improves conditioning for BFGS starting from all ones.
    
    Basis (10 params):
    p0: P0(x) = 1
    p1: P1(x) = 2*xn - 1
    p2: P2(x) = 6*xn^2 - 6*xn + 1
    p3: P3(x) = 20*xn^3 - 30*xn^2 + 12*xn - 1
    p4: P0(y) = 1 (redundant with p0, but keeps structure symmetric)
       Actually, let's use a more efficient allocation:
    
    Let me reconsider. We have 10 params. Let's use:
    p0: bias (P0 for all, just 1)
    p1: P1(x)
    p2: P1(y)
    p3: P1(z)
    p4: P2(x)
    p5: P2(y)
    p6: P2(z)
    p7: P3(x)
    p8: P3(y)
    p9: P1(x)*P1(y)  (key interaction)
    
    This gives us cubic-level approximation in each variable and one interaction.
    """
    # Check time budget
    if time.time() - _START_TIME > _TIME_LIMIT - _TIME_MARGIN:
        # Return a simple prediction as fallback
        return params[0] * np.ones_like(np.asarray(x, dtype=float))
    
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    p = np.asarray(params, dtype=float)
    
    # Normalize inputs to [0, 1] range
    # Original range is [0.5, 2], so subtract 0.5 and divide by 1.5
    xn = (x - 0.5) / 1.5
    yn = (y - 0.5) / 1.5
    zn = (z - 0.5) / 1.5
    
    # Shifted Legendre polynomials on [0, 1]
    # P0(t) = 1
    # P1(t) = 2t - 1
    # P2(t) = 6t^2 - 6t + 1
    # P3(t) = 20t^3 - 30t^2 + 12t - 1
    
    # For x
    P0_x = np.ones_like(xn)
    P1_x = 2.0 * xn - 1.0
    P2_x = 6.0 * xn**2 - 6.0 * xn + 1.0
    P3_x = 20.0 * xn**3 - 30.0 * xn**2 + 12.0 * xn - 1.0
    
    # For y
    P0_y = np.ones_like(yn)
    P1_y = 2.0 * yn - 1.0
    P2_y = 6.0 * yn**2 - 6.0 * yn + 1.0
    P3_y = 20.0 * yn**3 - 30.0 * yn**2 + 12.0 * yn - 1.0
    
    # For z
    P0_z = np.ones_like(zn)
    P1_z = 2.0 * zn - 1.0
    P2_z = 6.0 * zn**2 - 6.0 * zn + 1.0
    P3_z = 20.0 * zn**3 - 30.0 * zn**2 + 12.0 * zn - 1.0
    
    # Build prediction
    # p0: bias
    # p1: P1(x)
    # p2: P1(y)
    # p3: P1(z)
    # p4: P2(x)
    # p5: P2(y)
    # p6: P2(z)
    # p7: P3(x)
    # p8: P3(y)
    # p9: P1(x)*P1(y)
    
    pred = (p[0] * P0_x +
            p[1] * P1_x +
            p[2] * P1_y +
            p[3] * P1_z +
            p[4] * P2_x +
            p[5] * P2_y +
            p[6] * P2_z +
            p[7] * P3_x +
            p[8] * P3_y +
            p[9] * (P1_x * P1_y))
    
    return pred
# EVOLVE-BLOCK-END
