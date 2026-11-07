import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression using a tensor-product Chebyshev basis.
    
    Basis terms (10 total):
    0: T0(x)T0(y)
    1: T1(x)T0(y)
    2: T2(x)T0(y)
    3: T0(x)T1(y)
    4: T1(x)T1(y)
    5: T2(x)T1(y)
    6: T0(x)T2(y)
    7: T1(x)T2(y)
    8: T2(x)T2(y)
    9: T3(x)T0(y)
    
    This structure prioritizes the square interaction space up to degree 2 in both
    variables, adding a cubic term in x. This differs from the total-degree-3
    triangular basis by including T2(x)T2(y) (degree 4) and excluding T0(x)T3(y).
    """
    # Calculate Chebyshev polynomials T0 to T3
    T0_x = np.ones_like(x)
    T1_x = x
    T2_x = 2.0 * x**2 - 1.0
    T3_x = 4.0 * x**3 - 3.0 * x

    T0_y = np.ones_like(y)
    T1_y = y
    T2_y = 2.0 * y**2 - 1.0
    T3_y = 4.0 * y**3 - 3.0 * y

    # Construct the 10 basis functions
    # Tensor product grid 3x3 (indices 0-2 for both) plus T3(x)
    f0 = T0_x * T0_y
    f1 = T1_x * T0_y
    f2 = T2_x * T0_y
    
    f3 = T0_x * T1_y
    f4 = T1_x * T1_y
    f5 = T2_x * T1_y
    
    f6 = T0_x * T2_y
    f7 = T1_x * T2_y
    f8 = T2_x * T2_y
    
    f9 = T3_x * T0_y

    # Combine with parameters
    basis = np.column_stack((
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9
    ))
    
    prediction = basis @ params
    
    return prediction
