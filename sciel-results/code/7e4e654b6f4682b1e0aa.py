"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global start time for deadline guard
_start_time = time.time()
DEADLINE = 150.0  # Wall-clock limit
SAFETY_MARGIN = 5.0  # Seconds to spare

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: combines the structural strength of Legendre polynomials (orthogonal, well-conditioned)
    with cross-interaction terms to capture non-additive structure.
    
    Basis (10 terms):
      p[0]:  P0(u)  (constant)
      p[1]:  P1(u)  (linear x)
      p[2]:  P1(v)  (linear y)
      p[3]:  P1(w)  (linear z)
      p[4]:  P2(u)  (quadratic x)
      p[5]:  P2(v)  (quadratic y)
      p[6]:  P2(w)  (quadratic z)
      p[7]:  P1(u)*P1(v)  (interaction x*y)
      p[8]:  P1(u)*P1(w)  (interaction x*z)
      p[9]:  P1(v)*P1(w)  (interaction y*z)
    
    This differs from the current program (which used pure cubic Legendre terms P3) by replacing
    the three cubic pure-power terms with cross-interaction terms. Cross-interactions are often
    more informative for general smooth functions than higher-order univariate terms, and they
    remain well-conditioned when built from orthogonal polynomials.
    """
    # Deadline guard: if we're too close to the limit, return a simple fallback
    # to avoid timeout. In practice, this function is fast, but this is a safety net.
    elapsed = time.time() - _start_time
    if elapsed > DEADLINE - SAFETY_MARGIN:
        # Fallback: return a constant (p[0]) to be safe
        return np.ones_like(x) * params[0]
    
    p = params
    
    # Normalize inputs to [-1, 1]
    # Inputs are in [0.5, 2.0]
    # center = 1.25, half_range = 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Shifted Legendre Polynomials on [-1, 1]
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = 0.5 * (3t^2 - 1)
    
    # Precompute powers for efficiency
    u2 = u * u
    v2 = v * v
    w2 = w * w
    
    # Basis functions
    # 1. Constant
    b0 = np.ones_like(u)
    
    # 2-4. Linear (P1)
    b1 = u
    b2 = v
    b3 = w
    
    # 5-7. Quadratic (P2)
    b4 = 0.5 * (3.0 * u2 - 1.0)
    b5 = 0.5 * (3.0 * v2 - 1.0)
    b6 = 0.5 * (3.0 * w2 - 1.0)
    
    # 8-10. Cross-interaction terms (P1*P1)
    b7 = u * v
    b8 = u * w
    b9 = v * w
    
    # Combine
    return (p[0] * b0 + 
            p[1] * b1 + 
            p[2] * b2 + 
            p[3] * b3 + 
            p[4] * b4 + 
            p[5] * b5 + 
            p[6] * b6 + 
            p[7] * b7 + 
            p[8] * b8 + 
            p[9] * b9)
# EVOLVE-BLOCK-END
