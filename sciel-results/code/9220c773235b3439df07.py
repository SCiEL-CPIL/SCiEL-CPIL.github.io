import numpy as np
import time

DIMENSIONS = [4, 4, 4]

_DEADLINE = time.time() + 120.0
_MARGIN = 5.0

def _time_ok():
    return time.time() < _DEADLINE - _MARGIN

def construct():
    n, m, p = DIMENSIONS

    # Strassen decomposition for 2x2 matrices
    # U2, V2, W2 are 4x7 matrices
    # Rows correspond to A00, A01, A10, A11 (for U2)
    # Rows correspond to B00, B01, B10, B11 (for V2)
    # Rows correspond to C00, C01, C10, C11 (for W2)
    
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # A00
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ], dtype=int)
    
    rank = 49
    U = np.zeros((16, rank), dtype=int)
    V = np.zeros((16, rank), dtype=int)
    W = np.zeros((16, rank), dtype=int)
    
    # Kronecker product construction:
    # For 4x4 = (2x2) ⊗ (2x2), each index splits into block and entry
    # i = 2*i1 + i0, j = 2*j1 + j0, k = 2*k1 + k0
    # U[i,j] uses U2[i1,j1] ⊗ U2[i0,j0]
    # V[j,k] uses V2[j1,k1] ⊗ V2[j0,k0]
    # W[k,i] uses W2[k1,i1] ⊗ W2[k0,i0]
    
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            
            u_idx = i * 4 + j
            u_block_row = i1 * 2 + j1
            u_entry_row = i0 * 2 + j0
            
            # Kronecker: U[u_idx, alpha*7+beta] = U2[u_block_row, alpha] * U2[u_entry_row, beta]
            for alpha in range(7):
                for beta in range(7):
                    col = alpha * 7 + beta
                    U[u_idx, col] = U2[u_block_row, alpha] * U2[u_entry_row, beta]
    
    for j in range(4):
        for k in range(4):
            j1, j0 = divmod(j, 2)
            k1, k0 = divmod(k, 2)
            
            v_idx = j * 4 + k
            v_block_row = j1 * 2 + k1
            v_entry_row = j0 * 2 + k0
            
            for alpha in range(7):
                for beta in range(7):
                    col = alpha * 7 + beta
                    V[v_idx, col] = V2[v_block_row, alpha] * V2[v_entry_row, beta]
    
    for k in range(4):
        for i in range(4):
            k1, k0 = divmod(k, 2)
            i1, i0 = divmod(i, 2)
            
            w_idx = k * 4 + i
            w_block_row = k1 * 2 + i1
            w_entry_row = k0 * 2 + i0
            
            for alpha in range(7):
                for beta in range(7):
                    col = alpha * 7 + beta
                    W[w_idx, col] = W2[w_block_row, alpha] * W2[w_entry_row, beta]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
