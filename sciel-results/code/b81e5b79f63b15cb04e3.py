import numpy as np
import time

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # Rank 42 strategy:
    # A (2x4) = [A1, A2] (2x2 each)
    # B (4x6) = [B1; B2] (2x6 each)
    # B1 = [B11, B12, B13] (2x2 each)
    # B2 = [B21, B22, B23] (2x2 each)
    # C = A1*B11 + A1*B12 + A1*B13 + A2*B21 + A2*B22 + A2*B23
    # Each term is 2x2x2 mult, rank 7.
    # Total rank 6 * 7 = 42.
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    def add_strassen_block(u_indices, v_indices, w_indices, start_col):
        # u_indices: [a, b, c, d] flat indices in U
        # v_indices: [e, f, g, h] flat indices in V
        # w_indices: [c11, c12, c21, c22] flat indices in W
        # Strassen formulas:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Column 0: M1
        U[u_indices[0], start_col] += 1
        U[u_indices[3], start_col] += 1
        V[v_indices[0], start_col] += 1
        V[v_indices[3], start_col] += 1
        W[w_indices[0], start_col] += 1
        W[w_indices[3], start_col] += 1
        
        # Column 1: M2
        U[u_indices[2], start_col+1] += 1
        U[u_indices[3], start_col+1] += 1
        V[v_indices[0], start_col+1] += 1
        W[w_indices[2], start_col+1] += 1
        
        # Column 2: M3
        U[u_indices[0], start_col+2] += 1
        V[v_indices[1], start_col+2] += 1
        V[v_indices[3], start_col+2] += -1
        W[w_indices[1], start_col+2] += 1
        W[w_indices[3], start_col+2] += 1
        
        # Column 3: M4
        U[u_indices[3], start_col+3] += 1
        V[v_indices[2], start_col+3] += 1
        V[v_indices[0], start_col+3] += -1
        W[w_indices[0], start_col+3] += 1
        W[w_indices[2], start_col+3] += 1
        
        # Column 4: M5
        U[u_indices[0], start_col+4] += 1
        U[u_indices[1], start_col+4] += 1
        V[v_indices[3], start_col+4] += 1
        W[w_indices[0], start_col+4] += -1
        W[w_indices[1], start_col+4] += 1
        
        # Column 5: M6
        U[u_indices[2], start_col+5] += 1
        U[u_indices[0], start_col+5] += -1
        V[v_indices[0], start_col+5] += 1
        V[v_indices[1], start_col+5] += 1
        W[w_indices[3], start_col+5] += 1
        
        # Column 6: M7
        U[u_indices[1], start_col+6] += 1
        U[u_indices[3], start_col+6] += -1
        V[v_indices[2], start_col+6] += 1
        V[v_indices[3], start_col+6] += 1
        W[w_indices[0], start_col+6] += 1

    # Indices in U (A is 2x4)
    # A = [[A00, A01, A02, A03],
    #      [A10, A11, A12, A13]]
    # Flat: 0,1,2,3, 4,5,6,7
    # A1 = [[A00, A01], [A10, A11]] -> Indices 0,1,4,5
    # A2 = [[A02, A03], [A12, A13]] -> Indices 2,3,6,7
    
    u_A1 = [0, 1, 4, 5]
    u_A2 = [2, 3, 6, 7]
    
    # Indices in V (B is 4x6)
    # B = [[B00..B05],
    #      [B10..B15],
    #      [B20..B25],
    #      [B30..B35]]
    # Flat: 0-5, 6-11, 12-17, 18-23
    
    # B1 = [[B00..B05], [B10..B15]]
    # B11 = [[B00, B01], [B10, B11]] -> Indices 0,1,6,7
    # B12 = [[B02, B03], [B12, B13]] -> Indices 2,3,8,9
    # B13 = [[B04, B05], [B14, B15]] -> Indices 4,5,10,11
    
    v_B11 = [0, 1, 6, 7]
    v_B12 = [2, 3, 8, 9]
    v_B13 = [4, 5, 10, 11]
    
    # B2 = [[B20..B25], [B30..B35]]
    # B21 = [[B20, B21], [B30, B31]] -> Indices 12,13,18,19
    # B22 = [[B22, B23], [B32, B33]] -> Indices 14,15,20,21
    # B23 = [[B24, B25], [B34, B35]] -> Indices 16,17,22,23
    
    v_B21 = [12, 13, 18, 19]
    v_B22 = [14, 15, 20, 21]
    v_B23 = [16, 17, 22, 23]
    
    # Indices in W (C is 2x6, stored as k*n+i? No, prompt says W[k*n+i])
    # Prompt: W[k*n+i, r].
    # C is 2x6. i in 0..1, k in 0..5.
    # Index = k*2 + i.
    # C = [[C00, C01, C02, C03, C04, C05],
    #      [C10, C11, C12, C13, C14, C15]]
    
    # Term A1*B11 produces C[:, 0:2]
    # C00, C01, C10, C11
    # C00 index: k=0, i=0 -> 0*2+0=0
    # C01 index: k=1, i=0 -> 1*2+0=2
    # C10 index: k=0, i=1 -> 0*2+1=1
    # C11 index: k=1, i=1 -> 1*2+1=3
    # So w_indices for C[:, 0:2] are [0, 2, 1, 3] corresponding to [C00, C01, C10, C11]
    # Strassen w_indices order: [C11, C12, C21, C22] in local 2x2 coords.
    # Local C11 -> Global C00 (0)
    # Local C12 -> Global C01 (2)
    # Local C21 -> Global C10 (1)
    # Local C22 -> Global C11 (3)
    
    w_C11 = [0, 2, 1, 3]
    
    # Term A1*B12 produces C[:, 2:4]
    # C02 (k=2, i=0) -> 4
    # C03 (k=3, i=0) -> 6
    # C12 (k=2, i=1) -> 5
    # C13 (k=3, i=1) -> 7
    w_C12 = [4, 6, 5, 7]
    
    # Term A1*B13 produces C[:, 4:6]
    # C04 (k=4, i=0) -> 8
    # C05 (k=5, i=0) -> 10
    # C14 (k=4, i=1) -> 9
    # C15 (k=5, i=1) -> 11
    w_C13 = [8, 10, 9, 11]
    
    # Term A2*B21 produces C[:, 0:2] (adds to it)
    w_C21 = [0, 2, 1, 3]
    
    # Term A2*B22 produces C[:, 2:4]
    w_C22 = [4, 6, 5, 7]
    
    # Term A2*B23 produces C[:, 4:6]
    w_C23 = [8, 10, 9, 11]

    col = 0
    add_strassen_block(u_A1, v_B11, w_C11, col)
    col += 7
    add_strassen_block(u_A1, v_B12, w_C12, col)
    col += 7
    add_strassen_block(u_A1, v_B13, w_C13, col)
    col += 7
    add_strassen_block(u_A2, v_B21, w_C21, col)
    col += 7
    add_strassen_block(u_A2, v_B22, w_C22, col)
    col += 7
    add_strassen_block(u_A2, v_B23, w_C23, col)
    col += 7
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

