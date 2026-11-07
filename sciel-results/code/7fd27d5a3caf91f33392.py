"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model:
    - Chebyshev polynomials T0-T3 for smooth trends.
    - Two sinusoidal components with learnable frequencies for periodic trends.
    """
    # Scale input to [-1, 1] for Chebyshev stability
    u = x / 3.0
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    
    a1s = params[4]
    a1c = params[5]
    w1_raw = params[6]
    
    a2s = params[7]
    a2c = params[8]
    w2_raw = params[9]
    
    # Stable softplus for frequency parameters to ensure positivity
    # softplus(z) = log(1 + exp(z))
    # For large z, softplus(z) ~ z. For small z, softplus(z) ~ log(2) + z/2.
    # We add 1.0 to ensure frequency is at least 1.0 when raw param is very negative.
    # Actually, softplus(-inf) is 0. So w = 1 + 0 = 1.
    # This ensures w >= 1.
    
    def softplus(z):
        # Numerical stability for exp(z)
        # If z > 20, exp(z) is large, log(1+exp(z)) ~ z
        # If z < -20, exp(z) is 0, log(1+0) ~ 0
        return np.where(z > 20, z, np.log1p(np.exp(z)))
    
    w1 = 1.0 + softplus(w1_raw)
    w2 = 1.0 + softplus(w2_raw)
    
    # Precompute Chebyshev polynomials
    # T0(u) = 1
    # T1(u) = u
    # T2(u) = 2u^2 - 1
    # T3(u) = 4u^3 - 3u
    
    T0 = np.ones_like(x)
    T1 = u
    T2 = 2.0 * u * u - 1.0
    T3 = 4.0 * u * u * u - 3.0 * u
    
    # Precompute sinusoidal terms
    sin1 = np.sin(w1 * x)
    cos1 = np.cos(w1 * x)
    sin2 = np.sin(w2 * x)
    cos2 = np.cos(w2 * x)
    
    # Linear combination
    y = c0 * T0 + c1 * T1 + c2 * T2 + c3 * T3
    y += a1s * sin1 + a1c * cos1
    y += a2s * sin2 + a2c * cos2
    
    return y
# EVOLVE-BLOCK-END
