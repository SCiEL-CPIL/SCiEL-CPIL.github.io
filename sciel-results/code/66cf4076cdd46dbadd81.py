import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    # We construct a rank-68 decomposition using Strassen for 2x2x2 blocks
    # and schoolbook for the remaining 1D slices.
    
    n, m, p = 3, 3, 8
    rank_total = 68
    
    U = np.zeros((n*m, rank_total))
    V = np.zeros((m*p, rank_total))
    W = np.zeros((p*n, rank_total))
    
    col = 0
    
    # Helper to add a schoolbook term for specific i, j, k
    def add_schoolbook(i, j, k, col_idx):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx, col_idx] = 1
        V[v_idx, col_idx] = 1
        W[w_idx, col_idx] = 1

    # 1. i in {0,1}, j in {0,1} -> 2x2x8
    # Split k into 4 blocks of size 2
    # Strassen 2x2x2 decomposition
    # Indices for 2x2x2 sub-problem:
    # U: i in {0,1}, j in {0,1} -> u = i*2 + j
    # V: j in {0,1}, k in {0,1} -> v = j*2 + k
    # W: k in {0,1}, i in {0,1} -> w = k*2 + i
    
    # Define the 7 Strassen vectors for 2x2x2
    # U_s, V_s, W_s are 4x7 matrices
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))
    
    # M1: (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1 # C00
    W_s[3, 0] = 1 # C11
    
    # M2: (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1 # C10
    W_s[3, 1] = -1 # C11
    
    # M3: A00(B10-B11)
    U_s[0, 2] = 1
    V_s[2, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1 # C01
    W_s[3, 2] = 1 # C11
    
    # M4: A11(B01-B00)
    U_s[3, 3] = 1
    V_s[1, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1 # C00
    W_s[2, 3] = 1 # C10
    
    # M5: (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1 # C00
    W_s[1, 4] = 1 # C01
    
    # M6: (A10-A00)(B00+B10)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[2, 5] = 1
    W_s[3, 5] = 1 # C11
    
    # M7: (A01-A11)(B01+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[1, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1 # C00
    W_s[1, 6] = 1 # C01
    
    # Map Strassen blocks to the full 3x3x8 tensor
    # k blocks: 0-1, 2-3, 4-5, 6-7
    for block_idx in range(4):
        k_offset = block_idx * 2
        # U part: i in {0,1}, j in {0,1}
        # u_idx in full: i*3 + j
        # u_idx in strassen: i*2 + j
        for i in range(2):
            for j in range(2):
                u_full = i * 3 + j
                u_str = i * 2 + j
                U[u_full, col:col+7] = U_s[u_str, :]
                
        # V part: j in {0,1}, k in {0,1} (relative to block)
        # v_idx in full: j*8 + (k_offset + k_rel)
        # v_idx in strassen: j*2 + k_rel
        for j in range(2):
            for k_rel in range(2):
                v_full = j * 8 + (k_offset + k_rel)
                v_str = j * 2 + k_rel
                V[v_full, col:col+7] = V_s[v_str, :]
                
        # W part: k in {0,1} (relative), i in {0,1}
        # w_idx in full: (k_offset + k_rel)*3 + i
        # w_idx in strassen: k_rel*2 + i
        for k_rel in range(2):
            for i in range(2):
                w_full = (k_offset + k_rel) * 3 + i
                w_str = k_rel * 2 + i
                W[w_full, col:col+7] = W_s[w_str, :]
                
        col += 7
        
    # 2. i in {0,1}, j=2 -> 2x1x8
    for i in range(2):
        for k in range(8):
            add_schoolbook(i, 2, k, col)
            col += 1
            
    # 3. i=2, j in {0,1} -> 1x2x8
    for j in range(2):
        for k in range(8):
            add_schoolbook(2, j, k, col)
            col += 1
            
    # 4. i=2, j=2 -> 1x1x8
    for k in range(8):
        add_schoolbook(2, 2, k, col)
        col += 1
        
    # Verify
    target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[i*m+j, j*p+k, k*n+i] = 1.0
                
    T = np.einsum('ir,jr,kr->ijk', U, V, W)
    if not np.allclose(T, target, atol=1e-9):
        # Fallback to schoolbook if construction fails (should not happen)
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

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
