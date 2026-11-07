import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    params = np.asarray(params, dtype=float)
    
    if len(params) < 10:
        params = np.pad(params, (0, 10 - len(params)))
    elif len(params) > 10:
        params = params[:10]
    
    # Bilinear Chebyshev basis: T_i(x) * T_j(x) for selected pairs
    # This allows capturing multiplicative interactions and sharp features
    # while remaining linear in the 10 coefficients (BFGS-friendly)
    
    # Compute Chebyshev polynomials up to degree 4
    # T0 = 1, T1 = x, T2 = 2x^2 - 1, T3 = 4x^3 - 3x, T4 = 8x^4 - 8x^2 + 1
    T0 = np.ones_like(x)
    T1 = x
    T2 = 2.0 * x**2 - 1.0
    T3 = 4.0 * x**3 - 3.0 * x
    T4 = 8.0 * x**4 - 8.0 * x**2 + 1.0
    
    # Select 10 bilinear basis functions:
    # (i,j) pairs: (0,0), (0,1), (0,2), (0,3), (0,4), (1,1), (1,2), (1,3), (2,2), (2,3)
    # These capture constant, linear, quadratic, cubic, quartic, and interaction terms
    
    basis = np.column_stack([
        T0 * T0,   # 1
        T0 * T1,   # x
        T0 * T2,   # 2x^2 - 1
        T0 * T3,   # 4x^3 - 3x
        T0 * T4,   # 8x^4 - 8x^2 + 1
        T1 * T1,   # x^2
        T1 * T2,   # x(2x^2 - 1) = 2x^3 - x
        T1 * T3,   # x(4x^3 - 3x) = 4x^4 - 3x^2
        T2 * T2,   # (2x^2 - 1)^2 = 4x^4 - 4x^2 + 1
        T2 * T3,   # (2x^2 - 1)(4x^3 - 3x)
    ])
    
    return basis @ params
