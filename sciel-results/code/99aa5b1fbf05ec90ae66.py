"""Matrix multiplication 3,3,4 tensor decomposition using block-structured Strassen."""
import numpy as np
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    start_time = time.time()
    deadline = start_time + 100  # safety margin before 150s kill
    
    # Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # i in [0,n), j in [0,m), k in [0,p)
    # U has shape (n*m, rank) = (9, rank)
    # V has shape (m*p, rank) = (12, rank)
    # W has shape (p*n, rank) = (12, rank)
    
    # Block structure:
    # A (3x3): rows 0-1 are "top", row 2 is "bottom"
    #           cols 0-1 are "left", col 2 is "right"
    # B (3x4): rows 0-1 are "top", row 2 is "bottom"
    #           cols 0-1 are "LL", col 2 is "LR", col 3 is "R"
    # C (3x4): rows 0-1 are "top", row 2 is "bottom"
    #           cols 0-1 are "LL", col 2 is "LR", col 3 is "R"
    
    # A blocks:
    # A11 = A[0:2, 0:2] (2x2)
    # A12 = A[0:2, 2]   (2x1)
    # A21 = A[2, 0:2]   (1x2)
    # A22 = A[2, 2]     (1x1)
    
    # B blocks:
    # B11 = B[0:2, 0:2] (2x2)
    # B12 = B[0:2, 2]   (2x1)
    # B13 = B[0:2, 3]   (2x1)
    # B21 = B[2, 0:2]   (1x2)
    # B22 = B[2, 2]     (1x1)
    # B23 = B[2, 3]     (1x1)
    
    # C blocks:
    # C11 = C[0:2, 0:2] (2x2) = A11*B11 + A12*B21
    # C12 = C[0:2, 2]   (2x1) = A11*B12 + A12*B22
    # C13 = C[0:2, 3]   (2x1) = A11*B13 + A12*B23
    # C21 = C[2, 0:2]   (1x2) = A21*B11 + A22*B21
    # C22 = C[2, 2]     (1x1) = A21*B12 + A22*B22
    # C23 = C[2, 3]     (1x1) = A21*B13 + A22*B23
    
    # We'll build the decomposition term by term.
    # Each term is a rank-1 tensor: U_col * V_col * W_col
    
    terms = []  # Each term: (U_entries, V_entries, W_entries) as dicts
    
    def add_term(u_dict, v_dict, w_dict):
        """Add a rank-1 term. u_dict maps (i,j)->val, v_dict maps (j,k)->val, w_dict maps (k,i)->val"""
        terms.append((u_dict, v_dict, w_dict))
    
    # Helper: create a 2x2 Strassen decomposition for given block indices
    # Strassen for multiplying A_block (2x2) by B_block (2x2) to get C_block (2x2)
    # A_block indices: (ai, aj) where ai,aj in {0,1} map to global (i,j)
    # B_block indices: (bj, bk) where bj in {0,1} maps to global row, bk in {0,1} maps to global col
    # C_block indices: (ci, ck) where ci in {0,1} maps to global row, ck in {0,1} maps to global col
    
    def strassen_2x2(ai_map, aj_map, bj_map, bk_map, ci_map, ck_map):
        """
        Add 7 Strassen terms for 2x2x2 multiplication.
        ai_map: (local_row, local_col) -> global (i, j) for A
        bj_map: (local_row, local_col) -> global (j, k) for B  
        ci_map: (local_row, local_col) -> global (k, i) for W
        """
        # Strassen products:
        # M1 = (A11+A22)*(B11+B22)
        # M2 = (A21+A22)*B11
        # M3 = A11*(B12-B22)
        # M4 = A22*(B21-B11)
        # M5 = (A11+A12)*B22
        # M6 = (A21-A11)*(B11+B12)
        # M7 = (A12-A22)*(B21+B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Local indices: A is 2x2 with indices (0,0),(0,1),(1,0),(1,1)
        # A11 = (0,0), A12 = (0,1), A21 = (1,0), A22 = (1,1)
        # B11 = (0,0), B12 = (0,1), B21 = (1,0), B22 = (1,1)
        # C11 = (0,0), C12 = (0,1), C21 = (1,0), C22 = (1,1)
        
        # M1: (A11+A22)*(B11+B22)
        u1 = {ai_map[0,0]: 1, ai_map[1,1]: 1}
        v1 = {bj_map[0,0]: 1, bj_map[1,1]: 1}
        w1 = {ci_map[0,0]: 1, ci_map[0,1]: 1, ci_map[1,0]: 1, ci_map[1,1]: 1}
        # C11 += M1, C22 += M1
        # Actually W should be: for each (k,i) in C, the coefficient
        # C11 gets +1, C12 gets +1, C21 gets +1, C22 gets +1
        # But W maps (k,i) -> value. For C[ci,ck], the W index is (k,i) = (ck, ci)
        # Let me be more careful.
        
        # W index is (k, i) where k is column of C, i is row of C.
        # So for C[ci, ck], W index is (ck, ci).
        
        # Let me redefine: W dict maps (k, i) -> value where k is C column, i is C row.
        
        # M1 contributes to all 4 C entries with coefficient +1
        w1 = {ci_map[0,0]: 1, ci_map[0,1]: 1, ci_map[1,0]: 1, ci_map[1,1]: 1}
        # But ci_map gives (k,i) for C[ci,ck]. So:
        # C[0,0] -> W index ci_map[0,0]
        # C[0,1] -> W index ci_map[0,1]
        # C[1,0] -> W index ci_map[1,0]
        # C[1,1] -> W index ci_map[1,1]
        add_term(u1, v1, w1)
        
        # M2: (A21+A22)*B11
        u2 = {ai_map[1,0]: 1, ai_map[1,1]: 1}
        v2 = {bj_map[0,0]: 1}
        # C21 += M2
        w2 = {ci_map[1,0]: 1}
        add_term(u2, v2, w2)
        
        # M3: A11*(B12-B22)
        u3 = {ai_map[0,0]: 1}
        v3 = {bj_map[0,1]: 1, bj_map[1,1]: -1}
        # C12 += M3, C22 += M3
        w3 = {ci_map[0,1]: 1, ci_map[1,1]: 1}
        add_term(u3, v3, w3)
        
        # M4: A22*(B21-B11)
        u4 = {ai_map[1,1]: 1}
        v4 = {bj_map[1,0]: 1, bj_map[0,0]: -1}
        # C11 += M4, C21 += M4
        w4 = {ci_map[0,0]: 1, ci_map[1,0]: 1}
        add_term(u4, v4, w4)
        
        # M5: (A11+A12)*B22
        u5 = {ai_map[0,0]: 1, ai_map[0,1]: 1}
        v5 = {bj_map[1,1]: 1}
        # C11 -= M5, C12 += M5
        w5 = {ci_map[0,0]: -1, ci_map[0,1]: 1}
        add_term(u5, v5, w5)
        
        # M6: (A21-A11)*(B11+B12)
        u6 = {ai_map[1,0]: 1, ai_map[0,0]: -1}
        v6 = {bj_map[0,0]: 1, bj_map[0,1]: 1}
        # C22 += M6
        w6 = {ci_map[1,1]: 1}
        add_term(u6, v6, w6)
        
        # M7: (A12-A22)*(B21+B22)
        u7 = {ai_map[0,1]: 1, ai_map[1,1]: -1}
        v7 = {bj_map[1,0]: 1, bj_map[1,1]: 1}
        # C11 += M7
        w7 = {ci_map[0,0]: 1}
        add_term(u7, v7, w7)
    
    # Define index mappings
    # A: (i, j) where i in [0,3), j in [0,3)
    # For A11 (2x2 top-left): local (r,c) -> global (r, c)
    # For A12 (2x1 top-right): local (r,0) -> global (r, 2)
    # For A21 (1x2 bottom-left): local (0,c) -> global (2, c)
    # For A22 (1x1 bottom-right): local (0,0) -> global (2, 2)
    
    # B: (j, k) where j in [0,3), k in [0,4)
    # For B11 (2x2): local (r,c) -> global (r, c)
    # For B12 (2x1): local (r,0) -> global (r, 2)
    # For B13 (2x1): local (r,0) -> global (r, 3)
    # For B21 (1x2): local (0,c) -> global (2, c)
    # For B22 (1x1): local (0,0) -> global (2, 2)
    # For B23 (1x1): local (0,0) -> global (2, 3)
    
    # C/W: (k, i) where k in [0,4), i in [0,3)
    # For C11 (2x2 top-left): local (r,c) -> global (c, r) i.e. W index (k=c, i=r)
    # For C12 (2x1 top-mid): local (r,0) -> global (2, r) i.e. W index (k=2, i=r)
    # For C13 (2x1 top-right): local (r,0) -> global (3, r) i.e. W index (k=3, i=r)
    # For C21 (1x2 bottom-left): local (0,c) -> global (c, 2) i.e. W index (k=c, i=2)
    # For C22 (1x1 bottom-mid): local (0,0) -> global (2, 2) i.e. W index (k=2, i=2)
    # For C23 (1x1 bottom-right): local (0,0) -> global (3, 2) i.e. W index (k=3, i=2)
    
    # C11: A11*B11 + A12*B21
    # A11*B11: 2x2 * 2x2 -> 2x2, use Strassen
    # A12*B21: (2x1)*(1x2) -> 2x2, rank 4 (each C entry is one product)
    
    # Strassen for A11*B11:
    # A11 local (r,c) -> global A index (r, c)
    # B11 local (r,c) -> global B index (r, c)
    # C11 local (r,c) -> global W index (k=c, i=r)
    ai_map_11 = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
    bj_map_11 = {(0,0): (0,0), (0,1): (0,1), (1,0): (1,0), (1,1): (1,1)}
    ci_map_11 = {(0,0): (0,0), (0,1): (1,0), (1,0): (0,1), (1,1): (1,1)}
    # Wait, ci_map should map local C (row, col) -> W index (k, i) = (col, row)
    # C11 local (r,c): r is row of C, c is col of C. W index = (k=c, i=r)
    ci_map_11 = {(0,0): (0,0), (0,1): (1,0), (1,0): (0,1), (1,1): (1,1)}
    strassen_2x2(ai_map_11, bj_map_11, ci_map_11)
    
    # A12*B21: (2x1)*(1x2) -> 2x2
    # A12: global (i,j) = (r, 2) for r in {0,1}
    # B21: global (j,k) = (2, c) for c in {0,1}
    # C11: global (k,i) = (c, r) for r,c in {0,1}
    # Each C[r,c] = A12[r] * B21[c]
    for r in range(2):
        for c in range(2):
            u = {(r, 2): 1}
            v = {(2, c): 1}
            w = {(c, r): 1}
            add_term(u, v, w)
    
    # C12: A11*B12 + A12*B22
    # A11*B12: (2x2)*(2x1) -> 2x1, rank 4
    # A12*B22: (2x1)*(1x1) -> 2x1, rank 2
    
    # A11*B12:
    # A11: global (i,j) = (r, c) for r,c in {0,1}
    # B12: global (j,k) = (r, 2) for r in {0,1}
    # C12: global (k,i) = (2, r) for r in {0,1}
    # C12[r] = sum_c A11[r,c] * B12[c]
    for r in range(2):
        for c in range(2):
            u = {(r, c): 1}
            v = {(c, 2): 1}
            w = {(2, r): 1}
            add_term(u, v, w)
    
    # A12*B22:
    # A12: global (i,j) = (r, 2) for r in {0,1}
    # B22: global (j,k) = (2, 2)
    # C12: global (k,i) = (2, r) for r in {0,1}
    # C12[r] = A12[r] * B22
    for r in range(2):
        u = {(r, 2): 1}
        v = {(2, 2): 1}
        w = {(2, r): 1}
        add_term(u, v, w)
    
    # C13: A11*B13 + A12*B23
    # A11*B13: (2x2)*(2x1) -> 2x1, rank 4
    # A12*B23: (2x1)*(1x1) -> 2x1, rank 2
    
    # A11*B13:
    # A11: global (i,j) = (r, c) for r,c in {0,1}
    # B13: global (j,k) = (r, 3) for r in {0,1}
    # C13: global (k,i) = (3, r) for r in {0,1}
    for r in range(2):
        for c in range(2):
            u = {(r, c): 1}
            v = {(c, 3): 1}
            w = {(3, r): 1}
            add_term(u, v, w)
    
    # A12*B23:
    # A12: global (i,j) = (r, 2) for r in {0,1}
    # B23: global (j,k) = (2, 3)
    # C13: global (k,i) = (3, r) for r in {0,1}
    for r in range(2):
        u = {(r, 2): 1}
        v = {(2, 3): 1}
        w = {(3, r): 1}
        add_term(u, v, w)
    
    # C21: A21*B11 + A22*B21
    # A21*B11: (1x2)*(2x2) -> 1x2, rank 4
    # A22*B21: (1x1)*(1x2) -> 1x2, rank 2
    
    # A21*B11:
    # A21: global (i,j) = (2, c) for c in {0,1}
    # B11: global (j,k) = (r, c) for r,c in {0,1}
    # C21: global (k,i) = (c, 2) for c in {0,1}
    # C21[c] = sum_r A21[r] * B11[r,c]
    for c in range(2):
        for r in range(2):
            u = {(2, r): 1}
            v = {(r, c): 1}
            w = {(c, 2): 1}
            add_term(u, v, w)
    
    # A22*B21:
    # A22: global (i,j) = (2, 2)
    # B21: global (j,k) = (2, c) for c in {0,1}
    # C21: global (k,i) = (c, 2) for c in {0,1}
    for c in range(2):
        u = {(2, 2): 1}
        v = {(2, c): 1}
        w = {(c, 2): 1}
        add_term(u, v, w)
    
    # C22: A21*B12 + A22*B22
    # A21*B12: (1x2)*(2x1) -> 1x1, rank 2
    # A22*B22: (1x1)*(1x1) -> 1x1, rank 1
    
    # A21*B12:
    # A21: global (i,j) = (2, r) for r in {0,1}
    # B12: global (j,k) = (r, 2) for r in {0,1}
    # C22: global (k,i) = (2, 2)
    # C22 = sum_r A21[r] * B12[r]
    for r in range(2):
        u = {(2, r): 1}
        v = {(r, 2): 1}
        w = {(2, 2): 1}
        add_term(u, v, w)
    
    # A22*B22:
    u = {(2, 2): 1}
    v = {(2, 2): 1}
    w = {(2, 2): 1}
    add_term(u, v, w)
    
    # C23: A21*B13 + A22*B23
    # A21*B13: (1x2)*(2x1) -> 1x1, rank 2
    # A22*B23: (1x1)*(1x1) -> 1x1, rank 1
    
    # A21*B13:
    # A21: global (i,j) = (2, r) for r in {0,1}
    # B13: global (j,k) = (r, 3) for r in {0,1}
    # C23: global (k,i) = (3, 2)
    for r in range(2):
        u = {(2, r): 1}
        v = {(r, 3): 1}
        w = {(3, 2): 1}
        add_term(u, v, w)
    
    # A22*B23:
    u = {(2, 2): 1}
    v = {(2, 3): 1}
    w = {(3, 2): 1}
    add_term(u, v, w)
    
    # Now build the matrices from terms
    rank = len(terms)
    U = np.zeros((n*m, rank), dtype=float)
    V = np.zeros((m*p, rank), dtype=float)
    W = np.zeros((p*n, rank), dtype=float)
    
    for c, (u_dict, v_dict, w_dict) in enumerate(terms):
        for (i, j), val in u_dict.items():
            U[i*m + j, c] += val
        for (j, k), val in v_dict.items():
            V[j*p + k, c] += val
        for (k, i), val in w_dict.items():
            W[k*n + i, c] += val
    
    # Verify the decomposition
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert to lists with integer/half-integer entries
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
