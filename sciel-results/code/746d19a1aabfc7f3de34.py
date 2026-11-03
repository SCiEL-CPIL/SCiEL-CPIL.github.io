import time
import numpy as np

DIMENSIONS = [2, 4, 6]
_DEADLINE_START = time.time()
_DEADLINE = _DEADLINE_START + 130

def _elapsed():
    return time.time() - _DEADLINE_START

def construct():
    n, m, p = DIMENSIONS
    
    # Start with the known rank-42 Strassen decomposition
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def add_strassen_block(i_start, j_start, k_start, offset):
        col = offset
        U[(i_start+0)*m + (j_start+0), col] += 1
        U[(i_start+1)*m + (j_start+1), col] += 1
        V[(j_start+0)*p + (k_start+0), col] += 1
        V[(j_start+1)*p + (k_start+1), col] += 1
        W[(k_start+0)*n + (i_start+0), col] += 1
        W[(k_start+1)*n + (i_start+1), col] += 1
        
        col = offset + 1
        U[(i_start+1)*m + (j_start+0), col] += 1
        U[(i_start+1)*m + (j_start+1), col] += 1
        V[(j_start+0)*p + (k_start+0), col] += 1
        W[(k_start+0)*n + (i_start+1), col] += 1
        W[(k_start+1)*n + (i_start+1), col] -= 1
        
        col = offset + 2
        U[(i_start+0)*m + (j_start+0), col] += 1
        V[(j_start+0)*p + (k_start+1), col] += 1
        V[(j_start+1)*p + (k_start+1), col] -= 1
        W[(k_start+1)*n + (i_start+0), col] += 1
        W[(k_start+1)*n + (i_start+1), col] += 1
        
        col = offset + 3
        U[(i_start+1)*m + (j_start+1), col] += 1
        V[(j_start+1)*p + (k_start+0), col] += 1
        V[(j_start+0)*p + (k_start+0), col] -= 1
        W[(k_start+0)*n + (i_start+0), col] += 1
        W[(k_start+0)*n + (i_start+1), col] += 1
        
        col = offset + 4
        U[(i_start+0)*m + (j_start+0), col] += 1
        U[(i_start+0)*m + (j_start+1), col] += 1
        V[(j_start+1)*p + (k_start+1), col] += 1
        W[(k_start+0)*n + (i_start+0), col] -= 1
        W[(k_start+1)*n + (i_start+0), col] += 1
        
        col = offset + 5
        U[(i_start+1)*m + (j_start+0), col] += 1
        U[(i_start+0)*m + (j_start+0), col] -= 1
        V[(j_start+0)*p + (k_start+0), col] += 1
        V[(j_start+0)*p + (k_start+1), col] += 1
        W[(k_start+1)*n + (i_start+1), col] += 1
        
        col = offset + 6
        U[(i_start+0)*m + (j_start+1), col] += 1
        U[(i_start+1)*m + (j_start+1), col] -= 1
        V[(j_start+1)*p + (k_start+0), col] += 1
        V[(j_start+1)*p + (k_start+1), col] += 1
        W[(k_start+0)*n + (i_start+0), col] += 1

    add_strassen_block(0, 0, 0, 0)
    add_strassen_block(0, 0, 2, 7)
    add_strassen_block(0, 0, 4, 14)
    add_strassen_block(0, 2, 0, 21)
    add_strassen_block(0, 2, 2, 28)
    add_strassen_block(0, 2, 4, 35)
    
    # Convert to list of lists for return
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
