import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # x is (N,)
    # params is (10,)
    
    # Construct basis matrix Phi of shape (N, 10)
    # Columns:
    # 0: 1
    # 1: x
    # 2: x^2
    # 3: x^3
    # 4: x^4
    # 5: sin(x)
    # 6: cos(x)
    # 7: sin(2x)
    # 8: cos(2x)
    # 9: exp(-x^2)
    
    phi = np.empty((x.shape[0], 10), dtype=np.float64)
    phi[:, 0] = 1.0
    phi[:, 1] = x
    phi[:, 2] = x**2
    phi[:, 3] = x**3
    phi[:, 4] = x**4
    phi[:, 5] = np.sin(x)
    phi[:, 6] = np.cos(x)
    phi[:, 7] = np.sin(2.0 * x)
    phi[:, 8] = np.cos(2.0 * x)
    phi[:, 9] = np.exp(-x**2)
    
    # y = phi @ params
    y = phi @ params
    
    return y
