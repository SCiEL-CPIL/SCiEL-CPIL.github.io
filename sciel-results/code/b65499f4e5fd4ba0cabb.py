import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Full Quadratic Polynomial Basis.
    
    Basis (10 terms):
    0: 1
    1: x
    2: y
    3: z
    4: x^2
    5: y^2
    6: z^2
    7: xy
    8: xz
    9: yz
    
    This model assumes the target function can be well-approximated by a polynomial
    of degree at most 2. This is a very common structure in symbolic regression
    benchmarks. It is linear in the parameters, ensuring reliable convergence
    with BFGS.
    """
    p = params
    
    # Compute basis terms
    # We use the raw inputs x, y, z which are in [0.5, 2]
    # This keeps the coefficients interpretable and the basis well-conditioned
    
    b0 = np.ones_like(x)
    b1 = x
    b2 = y
    b3 = z
    b4 = x * x
    b5 = y * y
    b6 = z * z
    b7 = x * y
    b8 = x * z
    b9 = y * z
    
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
