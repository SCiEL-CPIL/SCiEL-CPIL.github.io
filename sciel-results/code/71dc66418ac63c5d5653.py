import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
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
    
    # Compute Kronecker products
    # U shape: (16, 49)
    # V shape: (16, 49)
    # W shape: (16, 49)
    
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)
    
    # We need to permute rows and columns to match the required index ordering.
    # Required U row index: i*m + j = 4*i + j
    # i = 2*i1 + i0, j = 2*j1 + j0
    # index = 8*i1 + 4*i0 + 2*j1 + j0
    
    # Kronecker row index (from np.kron):
    # If A is (4,7) and B is (4,7), A@B is (16, 49).
    # Row index R = r_A * 4 + r_B
    # r_A is row in U2 (block level), r_B is row in U2 (entry level)
    # r_A = i1*2 + j1
    # r_B = i0*2 + j0
    # R = (i1*2 + j1)*4 + (i0*2 + j0) = 8*i1 + 4*j1 + 2*i0 + j0
    
    # We need to map R -> index
    # Let's create a permutation array perm_row such that U_final[index] = U_kron[R]
    # Or U_final = U_kron[perm_row] where perm_row[index] = R
    
    perm_row_u = np.zeros(16, dtype=int)
    perm_col_u = np.zeros(49, dtype=int)
    
    for i in range(4):
        for j in range(4):
            i1, i0 = divmod(i, 2)
            j1, j0 = divmod(j, 2)
            target_idx = i * 4 + j
            # Source index in kron matrix
            r_A = i1 * 2 + j1
            r_B = i0 * 2 + j0
            src_idx = r_A * 4 + r_B
            perm_row_u[target_idx] = src_idx
            
    for c1 in range(7):
        for c2 in range(7):
            target_idx = c1 * 7 + c2
            # For columns, the structure is similar.
            # Column in U_kron is c_A * 7 + c_B
            # c_A corresponds to block col, c_B to entry col.
            # In the tensor decomposition, the column index c represents the "rank" index.
            # The rank index is formed by (alpha, beta) where alpha is from the first Strassen step and beta from the second.
            # In the Kronecker product, the column index is c_A * 7 + c_B.
            # This matches the target index c1 * 7 + c2 directly if we identify c1=c_A and c2=c_B.
            # So no permutation is needed for columns?
            # Let's verify.
            # U[i,j, c1, c2] = U2[i1,j1,c1] * U2[i0,j0,c2].
            # np.kron(U2, U2)[r_A*4+r_B, c_A*7+c_B] = U2[r_A, c_A] * U2[r_B, c_B].
            # Yes, the column index is already in the correct order (c1*7+c2).
            perm_col_u[target_idx] = target_idx
            
    # Apply permutation to rows
    U = U_kron[perm_row_u, :]
    
    # For V:
    # Row index: j*p + k = 4*j + k
    # j = 2*j1 + j0, k = 2*k1 + k0
    # index = 8*j1 + 4*j0 + 2*k1 + k0
    
    # Kronecker row index for V_kron (V2 @ V2):
    # r_A = j1*2 + k1 (block)
    # r_B = j0*2 + k0 (entry)
    # src_idx = r_A * 4 + r_B = 8*j1 + 4*k1 + 2*j0 + k0
    
    perm_row_v = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            j1, j0 = divmod(j, 2)
            k1, k0 = divmod(k, 2)
            target_idx = j * 4 + k
            r_A = j1 * 2 + k1
            r_B = j0 * 2 + k0
            src_idx = r_A * 4 + r_B
            perm_row_v[target_idx] = src_idx
            
    V = V_kron[perm_row_v, :]
    
    # For W:
    # Row index: k*n + i = 4*k + i
    # k = 2*k1 + k0, i = 2*i1 + i0
    # index = 8*k1 + 4*k0 + 2*i1 + i0
    
    # Kronecker row index for W_kron (W2 @ W2):
    # r_A = k1*2 + i1 (block)
    # r_B = k0*2 + i0 (entry)
    # src_idx = r_A * 4 + r_B = 8*k1 + 4*i1 + 2*k0 + i0
    
    perm_row_w = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            k1, k0 = divmod(k, 2)
            i1, i0 = divmod(i, 2)
            target_idx = k * 4 + i
            r_A = k1 * 2 + i1
            r_B = k0 * 2 + i0
            src_idx = r_A * 4 + r_B
            perm_row_w[target_idx] = src_idx
            
    W = W_kron[perm_row_w, :]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
