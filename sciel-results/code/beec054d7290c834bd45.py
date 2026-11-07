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
    # b0 = T0*T0 = 1 (constant)
    # b1 = T0*T1 = x (linear)
    # b2 = T0*T2 = 2x^2 - 1 (quadratic)
    # b3 = T1*T2 = x(2x^2-1) = 2x^3 - x (cubic, captures |x|-like features)
    # b4 = T0*T3 = 4x^3 - 3x (cubic)
    # b5 = T1*T3 = x(4x^3-3x) = 4x^4 - 3x^2 (quartic even)
    # b6 = T0*T4 = 8x^4 - 8x^2 + 1 (quartic)
    # b7 = T1*T4 = x(8x^4-8x^2+1) = 8x^5 - 8x^3 + x (quintic odd)
    # b8 = T2*T2 = (2x^2-1)^2 = 4x^4 - 4x^2 + 1 (quartic even, different from b6)
    # b9 = T2*T3 = (2x^2-1)(4x^3-3x) = 8x^5 - 10x^3 + 2x (quintic odd)
    
    b0 = T0
    b1 = T1
    b2 = T2
    b3 = T1 * T2
    b4 = T3
    b5 = T1 * T3
    b6 = T4
    b7 = T1 * T4
    b8 = T2 * T2
    b9 = T2 * T3
    
    # Build basis matrix: shape (n_points, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination with 10 parameters
    return basis @ params
