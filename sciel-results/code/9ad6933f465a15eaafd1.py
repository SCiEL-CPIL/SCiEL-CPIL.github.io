import numpy as np
import time
from itertools import combinations

DIMENSIONS = [3, 3, 6]
DEADLINE = 140.0  # Safety margin before 150s kill

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Get Laderman 3x3 decomposition
    U_3x3, V_3x3, W_3x3 = laderman_3x3()
    
    # Construct 3x3x6 decomposition using doubled Laderman (rank 46)
    U = np.zeros((9, 46))
    V = np.zeros((18, 46))
    W = np.zeros((18, 46))
    
    # First 23 columns correspond to B1 (cols 0-2 of B)
    U[:, 0:23] = U_3x3
    V[0:9, 0:23] = V_3x3
    W[0:9, 0:23] = W_3x3
    
    # Next 23 columns correspond to B2 (cols 3-5 of B)
    U[:, 23:46] = U_3x3
    V[9:18, 23:46] = V_3x3
    W[9:18, 23:46] = W_3x3
    
    # Verify 3x3x6 decomposition
    T_target_6 = np.zeros((9, 18, 18))
    for i in range(3):
        for j in range(3):
            for k in range(6):
                T_target_6[i*3+j, j*6+k, k*3+i] = 1.0
                
    T_calc_6 = np.zeros((9, 18, 18))
    for c in range(46):
        T_calc_6 += np.einsum('i,j,k->ijk', U[:,c], V[:,c], W[:,c])
        
    if not np.allclose(T_calc_6, T_target_6, atol=1e-9):
        return schoolbook_construct()
    
    # Try to reduce rank by finding linear dependencies
    if time.time() - start_time < DEADLINE - 30:
        U_reduced, V_reduced, W_reduced = try_reduce_rank(U, V, W, T_target_6)
        if U_reduced is not None:
            U, V, W = U_reduced, V_reduced, W_reduced
    
    if time.time() - start_time > DEADLINE:
        return schoolbook_construct()

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def try_reduce_rank(U, V, W, T_target):
    """Try to reduce the rank of the decomposition."""
    n_cols = U.shape[1]
    n, m, p = DIMENSIONS
    nm, mp, np_ = n * m, m * p, p * n
    
    # Try removing pairs of columns that might be linearly dependent
    for trial in range(5):
        if time.time() - start_time_global > DEADLINE - 10:
            break
        # Try to find if any column is a linear combination of others
        # For simplicity, try removing one column at a time and checking
        for c in range(n_cols):
            U_test = np.delete(U, c, axis=1)
            V_test = np.delete(V, c, axis=1)
            W_test = np.delete(W, c, axis=1)
            
            # Check if the remaining columns still span the target
            T_calc = np.zeros((nm, mp, np_))
            for cc in range(U_test.shape[1]):
                T_calc += np.einsum('i,j,k->ijk', U_test[:,cc], V_test[:,cc], W_test[:,cc])
            
            if np.allclose(T_calc, T_target, atol=1e-9):
                return U_test, V_test, W_test
    
    return None, None, None

def schoolbook_construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def laderman_3x3():
    U = np.array([
        [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
    ])
    V = np.array([
        [1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    ])
    W = np.array([
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    ])
    return U, V, W
