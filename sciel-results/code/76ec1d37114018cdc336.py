"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 6]

import time
import numpy as np

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication."""
    start_time = time.time()
    DEADLINE = 115
    
    n, m, p = DIMENSIONS  # 4, 5, 6
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    best_result = schoolbook()
    best_rank = n * m * p
    
    # Verify a decomposition
    def verify(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition terms
    # Each term is (u_dict, v_dict, w_dict)
    # u_dict keys: (i,j) for A[i,j], i,j in 0..1
    # v_dict keys: (j,k) for B[j,k], j,k in 0..1
    # w_dict keys: (k,i) for C[i,k] contribution, k,i in 0..1
    def strassen_2x2x2():
        # M1 = (A00+A11)*(B00+B11) -> C00 += 1, C11 += 1
        # M2 = (A00+A10)*B00 -> C10 += 1
        # M3 = A00*(B01-B11) -> C01 += 1
        # M4 = A11*(B10-B00) -> C00 += 1, C10 += 1
        # M5 = (A01+A11)*B11 -> C01 += 1, C11 -= 1
        # M6 = (A00-A10)*(B10+B11) -> C11 += 1
        # M7 = (A01-A11)*(B00+B10) -> C00 += 1
        
        return [
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(0,0): 1, (1,0): 1}, {(0,0): 1}, {(0,1): 1}),
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(1,0): 1}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
            # M5
            ({(0,1): 1, (1,1): 1}, {(1,1): 1}, {(1,0): 1, (1,1): -1}),
            # M6
            ({(0,0): 1, (1,0): -1}, {(1,0): 1, (1,1): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(0,0): 1, (1,0): 1}, {(0,0): 1})
        ]
    
    # Schoolbook for nxm x mxp
    def schoolbook_block(n, m, p):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = {(i,j): 1}
                    v = {(j,k): 1}
                    w = {(k,i): 1}
                    terms.append((u, v, w))
        return terms
    
    # Build global decomposition by combining block decompositions
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_block_terms(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset, terms):
        """
        Add block terms to global decomposition.
        a_row_offset, a_col_offset: offset in A (n x m)
        b_row_offset, b_col_offset: offset in B (m x p)
        c_row_offset, c_col_offset: offset in C (n x p)
        terms: list of (u_dict, v_dict, w_dict) with local indices
        """
        for u_dict, v_dict, w_dict in terms:
            u_global = [0.0] * (n * m)
            v_global = [0.0] * (m * p)
            w_global = [0.0] * (p * n)
            
            for (li, lj), val in u_dict.items():
                gi = a_row_offset + li
                gj = a_col_offset + lj
                u_global[gi * m + gj] = val
            
            for (lj, lk), val in v_dict.items():
                gj = b_row_offset + lj
                gk = b_col_offset + lk
                v_global[gj * p + gk] = val
            
            for (lk, li), val in w_dict.items():
                gk = c_col_offset + lk
                gi = c_row_offset + li
                w_global[gk * n + gi] = val
            
            U_cols.append(u_global)
            V_cols.append(v_global)
            W_cols.append(w_global)
    
    # Block decomposition of 4x5 * 5x6:
    # A (4x5): split rows 2+2, cols 2+3
    #   A11: rows 0-1, cols 0-1 (2x2)
    #   A12: rows 0-1, cols 2-4 (2x3)
    #   A21: rows 2-3, cols 0-1 (2x2)
    #   A22: rows 2-3, cols 2-4 (2x3)
    # B (5x6): split rows 2+3, cols 2+2+2 (or 3+3)
    #   Let's use cols 3+3: B1: rows 0-1, cols 0-2 (2x3); B2: rows 0-1, cols 3-5 (2x3)
    #   B3: rows 2-4, cols 0-2 (3x3); B4: rows 2-4, cols 3-5 (3x3)
    # C (4x6): split rows 2+2, cols 3+3
    #   C11: rows 0-1, cols 0-2 (2x3)
    #   C12: rows 0-1, cols 3-5 (2x3)
    #   C21: rows 2-3, cols 0-2 (2x3)
    #   C22: rows 2-3, cols 3-5 (2x3)
    
    # C11 = A11*B1 + A12*B3
    # C12 = A11*B2 + A12*B4
    # C21 = A21*B1 + A22*B3
    # C22 = A21*B2 + A22*B4
    
    # For 2x2 * 2x3: split B into 2x2 + 2x1
    #   2x2*2x2: Strassen (7 terms)
    #   2x2*2x1: Schoolbook (4 terms)
    #   Total: 11 terms
    # For 2x3 * 3x3: Schoolbook (18 terms)
    
    # Helper: 2x2 * 2x3 decomposition
    def decomp_2x2_by_2x3():
        terms = []
        # 2x2 * first 2 cols of 2x3: Strassen
        # B11 (2x2) is cols 0-1 of the 2x3
        strassen_terms = strassen_2x2x2()
        for u, v, w in strassen_terms:
            # u keys: (i,j) j in 0..1 -> cols 0-1
            # v keys: (j,k) k in 0..1 -> cols 0-1
            # w keys: (k,i) k in 0..1 -> cols 0-1
            terms.append((u, v, w))
        
        # 2x2 * last 1 col of 2x3: Schoolbook
        for i in range(2):
            for j in range(2):
                u = {(i,j): 1}
                v = {(j,2): 1}  # col 2 of the 2x3
                w = {(2,i): 1}  # k=2, i
                terms.append((u, v, w))
        
        return terms
    
    # Helper: 2x3 * 3x3 decomposition (schoolbook)
    def decomp_2x3_by_3x3():
        return schoolbook_block(2, 3, 3)
    
    # Add all 8 block products
    # C11 = A11*B1 + A12*B3
    # A11 is at A rows 0-1, cols 0-1; B1 is at B rows 0-1, cols 0-2; C11 is at C rows 0-1, cols 0-2
    add_block_terms(0, 0, 0, 0, 0, 0, decomp_2x2_by_2x3())
    # A12 is at A rows 0-1, cols 2-4; B3 is at B rows 2-4, cols 0-2; C11 is at C rows 0-1, cols 0-2
    add_block_terms(0, 2, 2, 0, 0, 0, decomp_2x3_by_3x3())
    
    # C12 = A11*B2 + A12*B4
    # B2 is at B rows 0-1, cols 3-5; C12 is at C rows 0-1, cols 3-5
    add_block_terms(0, 0, 0, 3, 0, 3, decomp_2x2_by_2x3())
    # B4 is at B rows 2-4, cols 3-5; C12 is at C rows 0-1, cols 3-5
    add_block_terms(0, 2, 2, 3, 0, 3, decomp_2x3_by_3x3())
    
    # C21 = A21*B1 + A22*B3
    # A21 is at A rows 2-3, cols 0-1; C21 is at C rows 2-3, cols 0-2
    add_block_terms(2, 0, 0, 0, 2, 0, decomp_2x2_by_2x3())
    # A22 is at A rows 2-3, cols 2-4; B3 is at B rows 2-4, cols 0-2
    add_block_terms(2, 2, 2, 0, 2, 0, decomp_2x3_by_3x3())
    
    # C22 = A21*B2 + A22*B4
    # B2 is at B rows 0-1, cols 3-5; C22 is at C rows 2-3, cols 3-5
    add_block_terms(2, 0, 0, 3, 2, 3, decomp_2x2_by_2x3())
    # B4 is at B rows 2-4, cols 3-5; C22 is at C rows 2-3, cols 3-5
    add_block_terms(2, 2, 2, 3, 2, 3, decomp_2x3_by_3x3())
    
    # Build result matrices
    rank = len(U_cols)
    U = [[U_cols[c][i] for c in range(rank)] for i in range(n * m)]
    V = [[V_cols[c][j] for c in range(rank)] for j in range(m * p)]
    W = [[W_cols[c][k] for c in range(rank)] for k in range(p * n)]
    
    result = {"U": U, "V": V, "W": W}
    
    # Verify
    if verify(U, V, W):
        if rank < best_rank:
            best_result = result
            best_rank = rank
    else:
        # Fall back to schoolbook
        best_result = schoolbook()
    
    return best_result

# EVOLVE-BLOCK-END
