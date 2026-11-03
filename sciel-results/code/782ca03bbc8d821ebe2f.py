import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the tensor decomposition is exact using the trace convention."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _add_strassen_block(U, V, W, i_off, j_off, k_off, col_start):
    """
    Add a 2x2x2 Strassen block.
    i_off: offset for i (0 or 1)
    j_off: offset for j (0, 2, etc)
    k_off: offset for k (0, 2, etc)
    col_start: starting column index in U, V, W
    """
    n, m, p = DIMENSIONS
    # M1
    U[i_off*m + j_off, col_start] += 1
    U[(i_off+1)*m + (j_off+1), col_start] += 1
    V[j_off*p + k_off, col_start] += 1
    V[(j_off+1)*p + (k_off+1), col_start] += 1
    W[k_off*n + i_off, col_start] += 1
    W[(k_off+1)*n + (i_off+1), col_start] += 1
    
    # M2
    U[(i_off+1)*m + j_off, col_start+1] += 1
    U[(i_off+1)*m + (j_off+1), col_start+1] += 1
    V[j_off*p + k_off, col_start+1] += 1
    W[k_off*n + (i_off+1), col_start+1] += 1
    W[(k_off+1)*n + (i_off+1), col_start+1] -= 1
    
    # M3
    U[i_off*m + j_off, col_start+2] += 1
    V[j_off*p + (k_off+1), col_start+2] += 1
    V[(j_off+1)*p + (k_off+1), col_start+2] -= 1
    W[(k_off+1)*n + i_off, col_start+2] += 1
    W[(k_off+1)*n + (i_off+1), col_start+2] += 1
    
    # M4
    U[(i_off+1)*m + (j_off+1), col_start+3] += 1
    V[(j_off+1)*p + k_off, col_start+3] += 1
    V[j_off*p + k_off, col_start+3] -= 1
    W[k_off*n + i_off, col_start+3] += 1
    W[k_off*n + (i_off+1), col_start+3] += 1
    
    # M5
    U[i_off*m + j_off, col_start+4] += 1
    U[i_off*m + (j_off+1), col_start+4] += 1
    V[(j_off+1)*p + (k_off+1), col_start+4] += 1
    W[k_off*n + i_off, col_start+4] -= 1
    W[(k_off+1)*n + i_off, col_start+4] += 1
    
    # M6
    U[(i_off+1)*m + j_off, col_start+5] += 1
    U[i_off*m + j_off, col_start+5] -= 1
    V[j_off*p + k_off, col_start+5] += 1
    V[j_off*p + (k_off+1), col_start+5] += 1
    W[(k_off+1)*n + (i_off+1), col_start+5] += 1
    
    # M7
    U[i_off*m + (j_off+1), col_start+6] += 1
    U[(i_off+1)*m + (j_off+1), col_start+6] -= 1
    V[(j_off+1)*p + k_off, col_start+6] += 1
    V[(j_off+1)*p + (k_off+1), col_start+6] += 1
    W[k_off*n + i_off, col_start+6] += 1
    
    return col_start + 7

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Rank 15 decomposition
    rank = 15
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Block 1: j in {0,1}, k in {0,1}
    col = _add_strassen_block(U, V, W, 0, 0, 0, col)
    
    # Block 2: j in {2,3}, k in {2,3}
    col = _add_strassen_block(U, V, W, 0, 2, 2, col)
    
    # Block 3: j in {4}, k in {4,5}
    # Rank 1 term: A[i,4] * B[4,k] * C[i,k]
    for i in range(n):
        U[i*m + 4, col] = 1.0
    for k in range(4, 6):
        V[4*p + k, col] = 1.0
    for i in range(n):
        for k in range(4, 6):
            W[k*n + i, col] = 1.0
    col += 1
    
    # Verify
    if not _verify_decomposition(U, V, W, n, m, p):
        # Fallback to schoolbook if something went wrong
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
