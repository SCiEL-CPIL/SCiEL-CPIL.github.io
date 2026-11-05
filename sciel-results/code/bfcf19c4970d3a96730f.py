import numpy as np
import time

def construct() -> dict:
    """
    Returns the schoolbook decomposition for 3x3 by 3x5 matrix multiplication.
    Rank = 3 * 3 * 5 = 45.
    """
    n, m, p = 3, 3, 5
    
    # Dimensions
    dim_U = n * m  # 9
    dim_V = m * p  # 15
    dim_W = p * n  # 15
    rank = n * m * p  # 45
    
    U = np.zeros((dim_U, rank), dtype=float)
    V = np.zeros((dim_V, rank), dtype=float)
    W = np.zeros((dim_W, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
                
    # Convert to lists of lists as required by the interface (though numpy arrays are often accepted, 
    # the prompt implies a dict with 'U', 'V', 'W' which might be checked as lists or arrays. 
    # The seed used lists. I will return numpy arrays if the evaluator accepts them, 
    # but to be safe and match the "list of lists" style of the seed's return type 
    # (which was converted to lists of floats), I will return lists.
    # However, numpy arrays are more efficient to construct. Let's check the seed.
    # Seed: result = {"U": [[float(x) for x in row] for row in U], ...}
    # I will return lists of floats to be safe.
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {
        "U": U_list,
        "V": V_list,
        "W": W_list
    }
