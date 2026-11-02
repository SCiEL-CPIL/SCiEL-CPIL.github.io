import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank is 35
    rank = 35
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen's 7 products for 2x2 matrix multiplication
    # A indices: 0,1,2,3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # B indices for a 2x2 block: (0,0), (0,1), (1,0), (1,1)
    # C indices for a 2x2 block: (0,0), (0,1), (1,0), (1,1)
    
    # U vectors (coefficients for A)
    # M1: A00 + A11
    u1 = [1, 0, 0, 1]
    # M2: A10 + A11
    u2 = [0, 0, 1, 1]
    # M3: A00
    u3 = [1, 0, 0, 0]
    # M4: A11
    u4 = [0, 0, 0, 1]
    # M5: A00 + A01
    u5 = [1, 1, 0, 0]
    # M6: A10 - A00
    u6 = [-1, 0, 1, 0]
    # M7: A01 - A11
    u7 = [0, 1, 0, -1]
    
    U_vecs = [u1, u2, u3, u4, u5, u6, u7]
    
    # V vectors (coefficients for B)
    # For a block at column offset 2b, B indices are:
    # j=0, k=2b -> index 0*10 + 2b = 2b
    # j=0, k=2b+1 -> index 0*10 + 2b+1 = 2b+1
    # j=1, k=2b -> index 1*10 + 2b = 10+2b
    # j=1, k=2b+1 -> index 1*10 + 2b+1 = 10+2b+1
    
    # M1: B00 + B11
    # M2: B00
    # M3: B01 - B11
    # M4: B10 - B00
    # M5: B11
    # M6: B00 + B01
    # M7: B10 + B11
    
    # We define the non-zero indices and values for V for each M
    # V_vecs[b][r] will be a list of (index, value)
    
    # W vectors (coefficients for C)
    # C indices:
    # i=0, k=2b -> index 2b*2 + 0 = 4b
    # i=1, k=2b -> index 2b*2 + 1 = 4b+1
    # i=0, k=2b+1 -> index (2b+1)*2 + 0 = 4b+2
    # i=1, k=2b+1 -> index (2b+1)*2 + 1 = 4b+3
    
    # M1: C00 + C11
    # M2: C10 - C11
    # M3: C01 + C11
    # M4: C00 + C10
    # M5: -C00 + C01
    # M6: C11
    # M7: C00
    
    for b in range(5):
        col_offset = 7 * b
        # V indices
        v_idx_00 = 2 * b
        v_idx_01 = 2 * b + 1
        v_idx_10 = 10 + 2 * b
        v_idx_11 = 10 + 2 * b + 1
        
        # W indices
        w_idx_00 = 4 * b
        w_idx_10 = 4 * b + 1
        w_idx_01 = 4 * b + 2
        w_idx_11 = 4 * b + 3
        
        # M1
        r = 0
        U[:, col_offset + r] = u1
        V[v_idx_00, col_offset + r] = 1
        V[v_idx_11, col_offset + r] = 1
        W[w_idx_00, col_offset + r] = 1
        W[w_idx_11, col_offset + r] = 1
        
        # M2
        r = 1
        U[:, col_offset + r] = u2
        V[v_idx_00, col_offset + r] = 1
        W[w_idx_10, col_offset + r] = 1
        W[w_idx_11, col_offset + r] = -1
        
        # M3
        r = 2
        U[:, col_offset + r] = u3
        V[v_idx_01, col_offset + r] = 1
        V[v_idx_11, col_offset + r] = -1
        W[w_idx_01, col_offset + r] = 1
        W[w_idx_11, col_offset + r] = 1
        
        # M4
        r = 3
        U[:, col_offset + r] = u4
        V[v_idx_10, col_offset + r] = 1
        V[v_idx_00, col_offset + r] = -1
        W[w_idx_00, col_offset + r] = 1
        W[w_idx_10, col_offset + r] = 1
        
        # M5
        r = 4
        U[:, col_offset + r] = u5
        V[v_idx_11, col_offset + r] = 1
        W[w_idx_00, col_offset + r] = -1
        W[w_idx_01, col_offset + r] = 1
        
        # M6
        r = 5
        U[:, col_offset + r] = u6
        V[v_idx_00, col_offset + r] = 1
        V[v_idx_01, col_offset + r] = 1
        W[w_idx_11, col_offset + r] = 1
        
        # M7
        r = 6
        U[:, col_offset + r] = u7
        V[v_idx_10, col_offset + r] = 1
        V[v_idx_11, col_offset + r] = 1
        W[w_idx_00, col_offset + r] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
