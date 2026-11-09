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
    deadline = time.time() + 110  # Safety margin
    
    # Schoolbook rank
    rank0 = n * m * p
    
    # We will replace two 2x2x2 blocks with Strassen (rank 7 each)
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 2: i in {2,3}, j in {2,3}, k in {2,3}
    
    # New Rank = 120 - 8 - 8 + 7 + 7 = 118
    new_rank = 118
    
    U = np.zeros((n*m, new_rank))
    V = np.zeros((m*p, new_rank))
    W = np.zeros((p*n, new_rank))
    
    # We need to fill the columns.
    # Let's iterate through all i, j, k.
    # If (i,j,k) is in Block 1 or Block 2, we skip it in schoolbook and let Strassen handle it.
    # But Strassen handles the *entire* 2x2x2 block.
    # So we should NOT add schoolbook columns for any (i,j,k) in those blocks.
    
    col = 0
    block1_i = {0, 1}
    block1_j = {0, 1}
    block1_k = {0, 1}
    
    block2_i = {2, 3}
    block2_j = {2, 3}
    block2_k = {2, 3}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i in block1_i and j in block1_j and k in block1_k):
                    continue
                if (i in block2_i and j in block2_j and k in block2_k):
                    continue
                
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    # Now add Strassen for Block 1
    _strassen_2x2x2(U, V, W, [0, 1], [0, 1], [0, 1], col)
    col += 7
    
    # Now add Strassen for Block 2
    _strassen_2x2x2(U, V, W, [2, 3], [2, 3], [2, 3], col)
    col += 7
    
    # col should be 118
    if col != new_rank:
        # Fallback to schoolbook if something went wrong
        U = np.zeros((n*m, rank0))
        V = np.zeros((m*p, rank0))
        W = np.zeros((p*n, rank0))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, c] = 1
                    V[j*p + k, c] = 1
                    W[k*n + i, c] = 1
                    c += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify
    if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to schoolbook if verification fails
        U = np.zeros((n*m, rank0))
        V = np.zeros((m*p, rank0))
        W = np.zeros((p*n, rank0))
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
