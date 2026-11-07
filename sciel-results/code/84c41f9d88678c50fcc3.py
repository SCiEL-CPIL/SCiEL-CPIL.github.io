import numpy as np

def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Ensure x and y are 1D arrays
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Construct the design matrix for a full cubic polynomial
    # Basis functions:
    # 0: 1
    # 1: x
    # 2: y
    # 3: x^2
    # 4: xy
    # 5: y^2
    # 6: x^3
    # 7: x^2y
    # 8: xy^2
    # 9: y^3
    
    # Calculate powers
    x2 = x * x
    x3 = x2 * x
    y2 = y * y
    y3 = y2 * y
    
    # Build the matrix
    # Shape: (N, 10)
    phi = np.column_stack((
        np.ones_like(x),
        x,
        y,
        x2,
        x * y,
        y2,
        x3,
        x2 * y,
        x * y2,
        y3
    ))
    
    # Prediction
    return phi @ params
