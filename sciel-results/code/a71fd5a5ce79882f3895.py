"""Matrix multiplication 4,4,9 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct a tensor decomposition for 4x4 x 4x9 matrix multiplication.
    Uses Strassen for the 4x4x4 sub-problems and schoolbook for the 4x4x1 part.
    Returns {'U', 'V', 'W'} with shapes (16, rank), (36, rank), (36, rank).
    """
    n, m, p = DIMENSIONS  # 4, 4, 9
    nm = n * m  # 16
    mp = m * p  # 36
    pn = p * n  # 36
    
    terms = []  # list of (u, v, w) triples
    
    def add_term(u, v, w):
        """Add a term to the list."""
        terms.append((u, v, w))
    
    # Index helpers
    def u_idx(i, j): return i * 4 + j  # A is 4x4
    def v_idx(j, k): return j * p + k  # B is 4xp
    def w_idx(k, i): return k * 4 + i  # C is p x 4
    
    # Strassen for 2x2 x 2x2 (rank 7)
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def strassen_4x4x4(col_offset):
        """
        Add Strassen terms for a 4x4 x 4x4 sub-problem.
        col_offset: the starting column index in B (0 or 4)
        """
        # Block structure:
        # A = [[A00, A01], [A10, A11]] (each 2x2)
        # B = [[B00, B01], [B10, B11]] (each 2x2, columns col_offset to col_offset+3)
        # C = [[C00, C01], [C10, C11]] (each 2x2, rows col_offset to col_offset+3)
        
        # S1 = (A00+A11)*(B00+B11)
        # A_factor[di,dj] = A[di,dj] + A[di+2,dj+2]
        # B_factor[di,dj] = B[di,col_offset+dj] + B[di+2,col_offset+dj+2]
        # C: C[col_offset+di, di] += 1, C[col_offset+di+2, di+2] += 1
        u1 = {}
        v1 = {}
        w1 = {}
        for di in range(2):
            for dj in range(2):
                u1[u_idx(di, dj)] = 1
                u1[u_idx(di+2, dj+2)] = 1
                v1[v_idx(di, col_offset+dj)] = 1
                v1[v_idx(di+2, col_offset+dj+2)] = 1
                w1[w_idx(col_offset+di, di)] = 1
                w1[w_idx(col_offset+di+2, di+2)] = 1
        add_term(u1, v1, w1)
        
        # S2 = (A10+A11)*B00
        # A_factor[di,dj] = A[di+2,dj] + A[di+2,dj+2]
        # B_factor[di,dj] = B[di,col_offset+dj]
        # C: C[col_offset+di+2, di] += 1
        u2 = {}
        v2 = {}
        w2 = {}
        for di in range(2):
            for dj in range(2):
                u2[u_idx(di+2, dj)] = 1
                u2[u_idx(di+2, dj+2)] = 1
                v2[v_idx(di, col_offset+dj)] = 1
                w2[w_idx(col_offset+di+2, di)] = 1
        add_term(u2, v2, w2)
        
        # S3 = A00*(B10-B11)
        # A_factor[di,dj] = A[di,dj]
        # B_factor[di,dj] = B[di+2,col_offset+dj] - B[di+2,col_offset+dj+2]
        # C: C[col_offset+di, di+2] += 1
        u3 = {}
        v3 = {}
        w3 = {}
        for di in range(2):
            for dj in range(2):
                u3[u_idx(di, dj)] = 1
                v3[v_idx(di+2, col_offset+dj)] = 1
                v3[v_idx(di+2, col_offset+dj+2)] = -1
                w3[w_idx(col_offset+di, di+2)] = 1
        add_term(u3, v3, w3)
        
        # S4 = A11*(B01-B00)
        # A_factor[di,dj] = A[di+2,dj+2]
        # B_factor[di,dj] = B[di,col_offset+dj+2] - B[di,col_offset+dj]
        # C: C[col_offset+di, di] += 1, C[col_offset+di+2, di] += 1
        u4 = {}
        v4 = {}
        w4 = {}
        for di in range(2):
            for dj in range(2):
                u4[u_idx(di+2, dj+2)] = 1
                v4[v_idx(di, col_offset+dj+2)] = 1
                v4[v_idx(di, col_offset+dj)] = -1
                w4[w_idx(col_offset+di, di)] = 1
                w4[w_idx(col_offset+di+2, di)] = 1
        add_term(u4, v4, w4)
        
        # S5 = (A00+A01)*B11
        # A_factor[di,dj] = A[di,dj] + A[di,dj+2]
        # B_factor[di,dj] = B[di+2,col_offset+dj+2]
        # C: C[col_offset+di, di+2] += 1, C[col_offset+di, di] -= 1
        u5 = {}
        v5 = {}
        w5 = {}
        for di in range(2):
            for dj in range(2):
                u5[u_idx(di, dj)] = 1
                u5[u_idx(di, dj+2)] = 1
                v5[v_idx(di+2, col_offset+dj+2)] = 1
                w5[w_idx(col_offset+di, di+2)] = 1
                w5[w_idx(col_offset+di, di)] = -1
        add_term(u5, v5, w5)
        
        # S6 = (A10-A00)*(B00+B01)
        # A_factor[di,dj] = A[di+2,dj] - A[di,dj]
        # B_factor[di,dj] = B[di,col_offset+dj] + B[di,col_offset+dj+2]
        # C: C[col_offset+di+2, di+2] += 1
        u6 = {}
        v6 = {}
        w6 = {}
        for di in range(2):
            for dj in range(2):
                u6[u_idx(di+2, dj)] = 1
                u6[u_idx(di, dj)] = -1
                v6[v_idx(di, col_offset+dj)] = 1
                v6[v_idx(di, col_offset+dj+2)] = 1
                w6[w_idx(col_offset+di+2, di+2)] = 1
        add_term(u6, v6, w6)
        
        # S7 = (A01-A11)*(B10+B11)
        # A_factor[di,dj] = A[di,dj+2] - A[di+2,dj+2]
        # B_factor[di,dj] = B[di+2,col_offset+dj] + B[di+2,col_offset+dj+2]
        # C: C[col_offset+di, di] += 1, C[col_offset+di+2, di+2] += 1
        u7 = {}
        v7 = {}
        w7 = {}
        for di in range(2):
            for dj in range(2):
                u7[u_idx(di, dj+2)] = 1
                u7[u_idx(di+2, dj+2)] = -1
                v7[v_idx(di+2, col_offset+dj)] = 1
                v7[v_idx(di+2, col_offset+dj+2)] = 1
                w7[w_idx(col_offset+di, di)] = 1
                w7[w_idx(col_offset+di+2, di+2)] = 1
        add_term(u7, v7, w7)
    
    # Add Strassen terms for columns 0-3
    strassen_4x4x4(0)
    # Add Strassen terms for columns 4-7
    strassen_4x4x4(4)
    
    # Schoolbook for column 8 (4x4 x 4x1)
    for i in range(4):
        for j in range(4):
            u8 = {u_idx(i, j): 1.0}
            v8 = {v_idx(j, 8): 1.0}
            w8 = {w_idx(8, i): 1.0}
            add_term(u8, v8, w8)
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    for r, (u, v, w) in enumerate(terms):
        for i in range(nm):
            U[i][r] = u.get(i, 0.0)
        for j in range(mp):
            V[j][r] = v.get(j, 0.0)
        for k in range(pn):
            W[k][r] = w.get(k, 0.0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
