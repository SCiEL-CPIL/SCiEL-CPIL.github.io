import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Equation for symbolic regression with 2 inputs x, y.
    
    Uses a Chebyshev polynomial basis of total degree up to 3.
    This provides a well-conditioned, structured approximation space on [-1, 1].
    The model is linear in the 10 parameters, allowing BFGS to converge
    reliably to the optimal coefficients for this basis.
    
    Basis functions:
    1. T0(x)T0(y)
    2. T1(x)T0(y)
    3. T0(x)T1(y)
    4. T2(x)T0(y)
    5. T1(x)T1(y)
    6. T0(x)T2(y)
    7. T3(x)T0(y)
    8. T2(x)T1(y)
    9. T1(x)T2(y)
    10. T0(x)T3(y)
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    # Compute Chebyshev polynomials T0..T3 for x and y
    # T0(t) = 1
    # T1(t) = t
    # T2(t) = 2t^2 - 1
    # T3(t) = 4t^3 - 3t
    
    Tx0 = np.ones_like(x)
    Tx1 = x
    Tx2 = 2 * x**2 - 1
    Tx3 = 4 * x**3 - 3 * x
    
    Ty0 = np.ones_like(y)
    Ty1 = y
    Ty2 = 2 * y**2 - 1
    Ty3 = 4 * y**3 - 3 * y
    
    # Construct the 10 basis functions
    b0 = Tx0 * Ty0
    b1 = Tx1 * Ty0
    b2 = Tx0 * Ty1
    b3 = Tx2 * Ty0
    b4 = Tx1 * Ty1
    b5 = Tx0 * Ty2
    b6 = Tx3 * Ty0
    b7 = Tx2 * Ty1
    b8 = Tx1 * Ty2
    b9 = Tx0 * Ty3
    
    # Stack into a matrix (n_samples, 10)
    basis = np.column_stack([b0, b1, b2, b3, b4, b5, b6, b7, b8, b9])
    
    # Linear combination
    pred = basis @ p
    
    # Safety check for non-finite values
    if not np.all(np.isfinite(pred)):
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
    return pred
