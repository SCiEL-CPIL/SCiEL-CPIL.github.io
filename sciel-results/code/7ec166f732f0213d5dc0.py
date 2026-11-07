"""Symbolic regression: stress-strain of an aluminium rod."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Bivariate polynomial chaos expansion (Taylor series) in strain and temperature.
    
    Uses up to 10 fitted constants to construct a smooth, differentiable surface.
    The basis functions are monomials e^i * t^j where i+j <= 4 (total order 4),
    which gives 15 terms. However, we only have 10 parameters.
    
    To use exactly 10 parameters, we select a subset of the most important low-order
    terms that capture the likely physics:
    - Linear in e and t
    - Quadratic in e (plastic hardening)
    - Interaction e*t (temperature-dependent modulus)
    - Cubic in e (higher order hardening)
    - t^2 (temperature nonlinearity)
    
    Let's define the 10 basis functions:
    1. 1
    2. e
    3. t
    4. e^2
    5. e*t
    6. t^2
    7. e^3
    8. e^2*t
    9. e*t^2
    10. t^3
    
    This covers all monomials up to total degree 3, except e^4, e^3*t, etc.
    Actually, total degree 3 has 10 terms:
    i+j <= 3:
    (0,0), (1,0), (2,0), (3,0)
    (0,1), (1,1), (2,1)
    (0,2), (1,2)
    (0,3)
    Total = 4 + 3 + 2 + 1 = 10.
    
    Perfect. We use a complete bivariate polynomial of total degree 3.
    """
    e = strain
    t = temp
    p = params

    # Basis functions for total degree <= 3
    # Order: (i, j) where term is e^i * t^j
    # 1. (0,0) -> 1
    # 2. (1,0) -> e
    # 3. (2,0) -> e^2
    # 4. (3,0) -> e^3
    # 5. (0,1) -> t
    # 6. (1,1) -> e*t
    # 7. (2,1) -> e^2*t
    # 8. (0,2) -> t^2
    # 9. (1,2) -> e*t^2
    # 10. (0,3) -> t^3

    term1 = np.ones_like(e)
    term2 = e
    term3 = e**2
    term4 = e**3
    term5 = t
    term6 = e * t
    term7 = e**2 * t
    term8 = t**2
    term9 = e * t**2
    term10 = t**3

    return (p[0] * term1 +
            p[1] * term2 +
            p[2] * term3 +
            p[3] * term4 +
            p[4] * term5 +
            p[5] * term6 +
            p[6] * term7 +
            p[7] * term8 +
            p[8] * term9 +
            p[9] * term10)
# EVOLVE-BLOCK-END
