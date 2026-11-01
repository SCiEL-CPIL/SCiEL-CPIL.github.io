import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def check_decomposition(U, V, W, n, m, p):
    """Verify the tensor decomposition identity."""
    r = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_arr = np.array(U, dtype=float) if isinstance(U, list) else U
    V_arr = np.array(V, dtype=float) if isinstance(V, list) else V
    W_arr = np.array(W, dtype=float) if isinstance(W, list) else W
    
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.sum(U_arr[a * m + b] * V_arr[c * p + d] * W_arr[k * n + i])
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = 2, 3, 6
    
    # Rank 33 decomposition
    # 3 blocks of 2x3x2, each rank 11
    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Define the 11-term decomposition for 2x3x2
    # A: 2x3, B: 3x2, C: 2x2
    # Indices for A: i*3+j
    # Indices for B: j*2+k (local)
    # Indices for C: k*2+i (local)
    
    # Local Strassen for A[:, 0:2] * B[0:2, :]
    # A_sub: 0,1,3,4
    # B_sub: 0,1,2,3 (local indices for j=0,1)
    # C_sub: 0,1,2,3
    
    # Local extra terms for j=2
    # A_sub2: 2,5
    # B_sub2: 4,5 (local indices for j=2)
    
    # We will map these local indices to global indices for each block.
    
    for block in range(3):
        # Global B column offset
        b_col_offset = block * 2
        # Global C column offset
        c_col_offset = block * 2
        
        # Term indices in the global U, V, W
        base_term = block * 11
        
        # --- Strassen Part (7 terms) ---
        # P1: (A00+A11)(B00+B11)
        U[0, base_term + 0] = 1; U[4, base_term + 0] = 1
        # B local 0,1 -> Global B[0, b_col_offset], B[1, b_col_offset]
        V[0 * 6 + b_col_offset, base_term + 0] = 1
        V[1 * 6 + b_col_offset, base_term + 0] = 1
        # C local 0,1 -> Global C[b_col_offset, 0], C[b_col_offset, 1] ? No.
        # C index is k*n+i.
        # Local C 0 is k=0,i=0. Global k=b_col_offset, i=0.
        # Local C 1 is k=0,i=1. Global k=b_col_offset, i=1.
        # Local C 2 is k=1,i=0. Global k=b_col_offset+1, i=0.
        # Local C 3 is k=1,i=1. Global k=b_col_offset+1, i=1.
        
        W[b_col_offset * 2 + 0, base_term + 0] = 1 # C00
        W[(b_col_offset + 1) * 2 + 1, base_term + 0] = 1 # C11
        
        # P2: (A10+A11)B00
        U[3, base_term + 1] = 1; U[4, base_term + 1] = 1
        V[0 * 6 + b_col_offset, base_term + 1] = 1
        W[b_col_offset * 2 + 1, base_term + 1] = 1 # C10
        
        # P3: A00(B01-B11)
        U[0, base_term + 2] = 1
        V[0 * 6 + b_col_offset + 1, base_term + 2] = 1 # B01
        V[1 * 6 + b_col_offset + 1, base_term + 2] = -1 # B11
        W[b_col_offset * 2 + 1, base_term + 2] = 1 # C01
        W[(b_col_offset + 1) * 2 + 1, base_term + 2] = 1 # C11
        
        # P4: A11(B10-B00)
        U[4, base_term + 3] = 1
        V[1 * 6 + b_col_offset, base_term + 3] = 1 # B10
        V[0 * 6 + b_col_offset, base_term + 3] = -1 # B00
        W[b_col_offset * 2 + 0, base_term + 3] = 1 # C00
        W[b_col_offset * 2 + 1, base_term + 3] = 1 # C10
        
        # P5: (A00+A01)B11
        U[0, base_term + 4] = 1; U[1, base_term + 4] = 1
        V[1 * 6 + b_col_offset + 1, base_term + 4] = 1 # B11
        W[b_col_offset * 2 + 0, base_term + 4] = -1 # C00
        W[b_col_offset * 2 + 1, base_term + 4] = 1 # C01
        
        # P6: (A10-A00)(B00+B01)
        U[3, base_term + 5] = 1; U[0, base_term + 5] = -1
        V[0 * 6 + b_col_offset, base_term + 5] = 1 # B00
        V[0 * 6 + b_col_offset + 1, base_term + 5] = 1 # B01
        W[(b_col_offset + 1) * 2 + 1, base_term + 5] = 1 # C11
        
        # P7: (A01-A11)(B10+B11)
        U[1, base_term + 6] = 1; U[4, base_term + 6] = -1
        V[1 * 6 + b_col_offset, base_term + 6] = 1 # B10
        V[1 * 6 + b_col_offset + 1, base_term + 6] = 1 # B11
        W[b_col_offset * 2 + 0, base_term + 6] = 1 # C00
        W[(b_col_offset + 1) * 2 + 1, base_term + 6] = 1 # C11
        
        # --- Extra Part (4 terms) ---
        # P8: A02 * B20
        U[2, base_term + 7] = 1
        V[2 * 6 + b_col_offset, base_term + 7] = 1
        W[b_col_offset * 2 + 0, base_term + 7] = 1 # C00
        
        # P9: A02 * B21
        U[2, base_term + 8] = 1
        V[2 * 6 + b_col_offset + 1, base_term + 8] = 1
        W[b_col_offset * 2 + 1, base_term + 8] = 1 # C01
        
        # P10: A12 * B20
        U[5, base_term + 9] = 1
        V[2 * 6 + b_col_offset, base_term + 9] = 1
        W[b_col_offset * 2 + 1, base_term + 9] = 1 # C10
        
        # P11: A12 * B21
        U[5, base_term + 10] = 1
        V[2 * 6 + b_col_offset + 1, base_term + 10] = 1
        W[(b_col_offset + 1) * 2 + 1, base_term + 10] = 1 # C11

    if check_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
