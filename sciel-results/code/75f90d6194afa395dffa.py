import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = x.flatten()
    y = y.flatten()

    # Chebyshev polynomial basis of total degree up to 4 (10 terms)
    # This provides a well-conditioned, linear-in-parameters basis
    # that can approximate smooth bivariate functions on [-1, 1]^2.

    # Chebyshev polynomials of the first kind
    def T0(t):
        return np.ones_like(t)

    def T1(t):
        return t

    def T2(t):
        return 2.0 * t**2 - 1.0

    def T3(t):
        return 4.0 * t**3 - 3.0 * t

    def T4(t):
        return 8.0 * t**4 - 8.0 * t**2 + 1.0

    # Build tensor-product Chebyshev basis
    # Strategy: Drop pure cubic terms (T3*T0, T0*T3) 
    # and add pure quartic terms (T4*T0, T0*T4).
    # This prioritizes quadratic interactions and high-order pure effects
    # over odd-symmetric cubic effects, which may be less prevalent 
    # in smooth functions on [-1, 1]^2.
    
    tx0 = T0(x)
    tx1 = T1(x)
    tx2 = T2(x)
    tx4 = T4(x)
    
    ty0 = T0(y)
    ty1 = T1(y)
    ty2 = T2(y)
    ty4 = T4(y)

    # Build the 10 basis functions explicitly
    basis = [
        tx0 * ty0,
        tx1 * ty0,
        tx0 * ty1,
        tx2 * ty0,
        tx1 * ty1,
        tx0 * ty2,
        tx2 * ty1,
        tx1 * ty2,
        tx4 * ty0,
        tx0 * ty4
    ]

    # Stack into (N, 10) matrix
    features = np.column_stack(basis)

    # Normalize each column to unit L2 norm to improve BFGS conditioning
    col_norms = np.sqrt(np.sum(features**2, axis=0))
    col_norms[col_norms < 1e-12] = 1.0
    features = features / col_norms

    return features @ params
