import numpy as np
import time


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression candidate: shifted Legendre polynomial basis (degree 0-3)
    on normalized inputs, with cross-interaction terms.
    
    This uses orthogonal polynomials for better numerical conditioning than raw
    monomials, capturing higher-order curvature. The 10 parameters are:
    - 1 constant
    - 6 cubic Legendre terms: P1,P2,P3 for each of u,v,w
    - 3 cross-interaction terms: P1(u)*P1(v), P1(u)*P1(w), P1(v)*P1(w)
    
    Normalization: inputs in [0.5, 2.0] -> u,v,w in [-1, 1].
    """
    p = params
    
    # Normalize to [-1, 1]
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Compute shifted Legendre polynomials P1, P2, P3 on [-1,1]
    # P1(t) = t
    # P2(t) = 0.5*(3*t^2 - 1)
    # P3(t) = 0.5*(5*t^3 - 3*t)
    
    P1_u = u
    P2_u = 0.5 * (3.0 * u * u - 1.0)
    P3_u = 0.5 * (5.0 * u * u * u - 3.0 * u)
    
    P1_v = v
    P2_v = 0.5 * (3.0 * v * v - 1.0)
    P3_v = 0.5 * (5.0 * v * v * v - 3.0 * v)
    
    P1_w = w
    P2_w = 0.5 * (3.0 * w * w - 1.0)
    P3_w = 0.5 * (5.0 * w * w * w - 3.0 * w)
    
    # Cross-interaction terms
    cross_uv = P1_u * P1_v
    cross_uw = P1_u * P1_w
    cross_vw = P1_v * P1_w
    
    # Basis allocation:
    # p[0]: constant
    # p[1]: P1_u
    # p[2]: P2_u
    # p[3]: P3_u
    # p[4]: P1_v
    # p[5]: P2_v
    # p[6]: P3_v
    # p[7]: P1_w
    # p[8]: P2_w
    # p[9]: P3_w
    # Wait, that's 10 terms already without cross terms.
    
    # Let me reconsider the allocation to include cross terms.
    # We have 10 params total.
    # Option: 
    #   1 constant
    #   3 linear (P1 for each)
    #   3 quadratic (P2 for each)
    #   3 cross interactions (P1*P1 pairs)
    # That's 1+3+3+3 = 10. This drops P3 (cubic) terms.
    
    # Alternative:
    #   1 constant
    #   6: P1, P2, P3 for u; P1, P2, P3 for v (drop w's higher terms?)
    # This is asymmetric.
    
    # Better symmetric allocation with cross terms:
    #   p[0]: constant
    #   p[1]: P1_u, p[2]: P1_v, p[3]: P1_w (linear)
    #   p[4]: P2_u, p[5]: P2_v, p[6]: P2_w (quadratic)
    #   p[7]: P1_u*P1_v, p[8]: P1_u*P1_w, p[9]: P1_v*P1_w (interactions)
    
    # This captures up to quadratic curvature per variable plus linear interactions.
    # It's symmetric and well-conditioned.
    
    result = (p[0] +
              p[1] * P1_u + p[2] * P1_v + p[3] * P1_w +
              p[4] * P2_u + p[5] * P2_v + p[6] * P2_w +
              p[7] * cross_uv + p[8] * cross_uw + p[9] * cross_vw)
    
    return result
