import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    p = params
    
    # Basis functions
    phi = np.empty((10, x.size), dtype=np.float64)
    phi[0] = 1.0
    phi[1] = x
    phi[2] = x**2
    phi[3] = x**3
    phi[4] = x**4
    phi[5] = np.sin(x)
    phi[6] = np.cos(x)
    phi[7] = np.sin(2.0 * x)
    phi[8] = np.cos(2.0 * x)
    phi[9] = np.exp(-x**2)
    
    return phi.T @ p
