import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    t_start = time.time()
    
    n_points = len(x)
    if n_points == 0:
        return np.array([])
    
    # Map x from [-3, 3] to t in [-1, 1]
    t = x / 3.0
    
    # Build Chebyshev polynomial basis T_0(t), T_1(t), ..., T_9(t)
    # Using the recurrence: T_0=1, T_1=t, T_{k+1} = 2t*T_k - T_{k-1}
    T0 = np.ones_like(t)
    T1 = t.copy()
    
    basis = [T0, T1]
    for k in range(2, 10):
        Tk = 2.0 * t * basis[-1] - basis[-2]
        basis.append(Tk)
    
    # Stack into matrix: shape (10, n_points)
    basis_matrix = np.vstack(basis)
    
    # Linear combination
    result = np.einsum('i,ij->j', params, basis_matrix)
    
    return result
