import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Map x from [0.1, 6] to [-1, 1]
    # t = (2 * (x - 0.1) / (6 - 0.1)) - 1
    # t = (2 * (x - 0.1) / 5.9) - 1
    # t = (x - 0.1) * (2 / 5.9) - 1
    
    t = (x - 0.1) * (2.0 / 5.9) - 1.0
    
    # Compute Chebyshev polynomials T_0 ... T_9
    # T_0 = 1
    # T_1 = t
    # T_k = 2*t*T_{k-1} - T_{k-2}
    
    # We need 10 basis functions.
    # We can build a matrix B of shape (N, 10) where B[:, k] = T_k(t)
    
    T0 = np.ones_like(t)
    T1 = t
    
    basis = np.zeros((len(t), 10))
    basis[:, 0] = T0
    basis[:, 1] = T1
    
    if len(t) > 0:
        for k in range(2, 10):
            Tk = 2.0 * t * T1 - T0
            basis[:, k] = Tk
            T0, T1 = T1, Tk
            
    return basis @ params
