"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    rank = len(U[0]) if isinstance(U[0], list) else U.shape[1]
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank):
        return False
    if V_arr.shape != (m*p, rank):
        return False
    if W_arr.shape != (p*n, rank):
        return False
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i*m + j]
                v_val = V_arr[j*p + k]
                w_val = W_arr[k*n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_2x2x2(U, V, W, i_indices, j_indices, k_indices, col_offset):
    """
    Add Strassen decomposition for a 2x2x2 block to U, V, W.
    i_indices: [i0, i1]
    j_indices: [j0, j1]
    k_indices: [k0, k1]
    col_offset: starting column index in U, V, W
    """
    n, m, p = DIMENSIONS
    i0, i1 = i_indices
    j0, j1 = j_indices
    k0, k1 = k_indices
    
    # Local indices 0, 1 map to global indices
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    
    # M1
    U[i0*m + j0, col_offset] += 1
    U[i1*m + j1, col_offset] += 1
    V[j0*p + k0, col_offset] += 1
    V[j1*p + k1, col_offset] += 1
    W[k0*n + i0, col_offset] += 1
    W[k1*n + i1, col_offset] += 1
    
    # M2
    U[i1*m + j0, col_offset + 1] += 1
    U[i1*m + j1, col_offset + 1] += 1
    V[j0*p + k0, col_offset + 1] += 1
    W[k0*n + i1, col_offset + 1] += 1
    W[k1*n + i1, col_offset + 1] += -1
    
    # M3
    U[i0*m + j0, col_offset + 2] += 1
    V[j0*p + k1, col_offset + 2] += 1
    V[j1*p + k1, col_offset + 2] += -1
    W[k1*n + i0, col_offset + 2] += 1
    
    # M4
    U[i1*m + j1, col_offset + 3] += 1
    V[j1*p + k0, col_offset + 3] += 1
    V[j0*p + k0, col_offset + 3] += -1
    W[k0*n + i1, col_offset + 3] += 1
    
    # M5
    U[i0*m + j0, col_offset + 4] += 1
    U[i0*m + j1, col_offset + 4] += 1
    V[j1*p + k1, col_offset + 4] += 1
    W[k1*n + i0, col_offset + 4] += 1
    
    # M6
    U[i1*m + j0, col_offset + 5] += 1
    U[i0*m + j0, col_offset + 5] += -1
    V[j0*p + k0, col_offset + 5] += 1
    V[j0*p + k1, col_offset + 5] += 1
    W[k1*n + i1, col_offset + 5] += 1
    
    # M7
    U[i0*m + j1, col_offset + 6] += 1
    U[i1*m + j1, col_offset + 6] += -1
    V[j1*p + k0, col_offset + 6] += 1
    V[j1*p + k1, col_offset + 6] += 1
    W[k0*n + i0, col_offset + 6] += 1

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, c] = 1
                V[j*p + k, c] = 1
                W[k*n + i, c] = 1
                c += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
