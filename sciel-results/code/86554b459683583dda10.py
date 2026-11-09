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
    Hybrid model combining linear, interaction, and exponential basis functions.
    
    Rationale:
    - The linear model scored 1.0 (predicting mean), indicating non-linearity.
    - The centered quadratic model scored 7.39, which is worse than linear. This suggests
      either the function is not well-approximated by a quadratic polynomial, or the
      optimizer struggled with the quadratic basis.
    - We switch to a basis that includes:
        1. Linear terms (centered) for baseline trends.
        2. Interaction terms (xy, xz, yz) to capture multiplicative dependencies.
        3. Exponential terms (exp(x), exp(y), exp(z)) to capture non-polynomial growth.
    - Centering inputs around 1.25 (midpoint of [0.5, 2]) improves numerical conditioning.
    - This basis is more flexible than a pure polynomial for functions involving
      multiplication or exponential behavior.
    
    Basis (10 params):
    p0: 1 (bias)
    p1: x_c (centered x)
    p2: y_c (centered y)
    p3: z_c (centered z)
    p4: x_c * y_c
    p5: x_c * z_c
    p6: y_c * z_c
    p7: exp(x_c)
    p8: exp(y_c)
    p9: exp(z_c)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    
    # Center the inputs around the midpoint of [0.5, 2], which is 1.25
    x_c = x - 1.25
    y_c = y - 1.25
    z_c = z - 1.25
    
    p = params
    
    # Compute basis functions
    # Linear
    b1 = x_c
    b2 = y_c
    b3 = z_c
    
    # Interactions
    b4 = x_c * y_c
    b5 = x_c * z_c
    b6 = y_c * z_c
    
    # Exponentials (clipped to prevent overflow, though x_c is small)
    # x_c is in [-0.75, 0.75], so exp(x_c) is in [0.47, 2.11], safe.
    b7 = np.exp(x_c)
    b8 = np.exp(y_c)
    b9 = np.exp(z_c)
    
    # Build the prediction as a linear combination
    pred = (p[0] * np.ones_like(x) +
            p[1] * b1 +
            p[2] * b2 +
            p[3] * b3 +
            p[4] * b4 +
            p[5] * b5 +
            p[6] * b6 +
            p[7] * b7 +
            p[8] * b8 +
            p[9] * b9)
    
    return pred
# EVOLVE-BLOCK-END
