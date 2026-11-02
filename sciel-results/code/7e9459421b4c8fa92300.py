"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Block decomposition:
    # A (3x4) -> A_top (2x4), A_bot (1x4)
    # B (4x5) -> B_left (4x2), B_mid (4x2), B_right (4x1)
    # C (3x5) -> C_tl(2x2), C_tm(2x2), C_tr(2x1), C_bl(1x2), C_bm(1x2), C_br(1x1)
    
    # Ranks:
    # C_tl = A_top(2x4) x B_left(4x2): split into two 2x2x2, each rank 7 -> 14
    # C_tm = A_top(2x4) x B_mid(4x2): same -> 14
    # C_tr = A_top(2x4) x B_right(4x1): schoolbook 2*4*1 = 8
    # C_bl = A_bot(1x4) x B_left(4x2): schoolbook 1*4*2 = 8
    # C_bm = A_bot(1x4) x B_mid(4x2): schoolbook 1*4*2 = 8
    # C_br = A_bot(1x4) x B_right(4x1): schoolbook 1*4*1 = 4
    # Total: 14+14+8+8+8+4 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    offset = 0
    
    def add_strassen_2x2x2(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add a rank-7 Strassen decomposition for a 2x2 x 2x2 -> 2x2 block."""
        nonlocal offset
        # A is 2x2 at rows a_row_offset..a_row_offset+1, cols a_col_offset..a_col_offset+1
        # B is 2x2 at rows b_row_offset..b_row_offset+1, cols b_col_offset..b_col_offset+1
        # C is 2x2 at rows c_row_offset..c_row_offset+1, cols c_col_offset..c_col_offset+1
        
        # Flat indices for A: [a_row_offset*m+a_col_offset, a_row_offset*m+a_col_offset+1,
        #                      (a_row_offset+1)*m+a_col_offset, (a_row_offset+1)*m+a_col_offset+1]
        # Flat indices for B: [b_row_offset*p+b_col_offset, b_row_offset*p+b_col_offset+1,
        #                      (b_row_offset+1)*p+b_col_offset, (b_row_offset+1)*p+b_col_offset+1]
        # Flat indices for C: [c_row_offset*n+c_col_offset, c_row_offset*n+c_col_offset+1,
        #                      (c_row_offset+1)*n+c_col_offset, (c_row_offset+1)*n+c_col_offset+1]
        
        a_idx = [a_row_offset * m + a_col_offset, a_row_offset * m + a_col_offset + 1,
                 (a_row_offset + 1) * m + a_col_offset, (a_row_offset + 1) * m + a_col_offset + 1]
        b_idx = [b_row_offset * p + b_col_offset, b_row_offset * p + b_col_offset + 1,
                 (b_row_offset + 1) * p + b_col_offset, (b_row_offset + 1) * p + b_col_offset + 1]
        c_idx = [c_row_offset * n + c_col_offset, c_row_offset * n + c_col_offset + 1,
                 (c_row_offset + 1) * n + c_col_offset, (c_row_offset + 1) * n + c_col_offset + 1]
        
        # Strassen terms:
        # P1 = A[0] * (B[1] - B[3])
        # P2 = (A[0] + A[2]) * B[3]
        # P3 = (A[1] - A[0]) * B[0]
        # P4 = A[3] * (B[1] - B[3])
        # P5 = (A[0] + A[1]) * B[0]
        # P6 = (A[2] - A[0]) * (B[0] + B[3])
        # P7 = (A[1] + A[3]) * (B[1] - B[3])
        
        # C[0] = P5 + P4 - P6 + P7
        # C[1] = P1 + P5
        # C[2] = P2 + P3
        # C[3] = P1 - P4 + P2 + P6
        
        terms = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # P1: U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
            # P2: U=[1,0,1,0], V=[0,0,0,1], W=[0,0,1,1]
            ([1,0,1,0], [0,0,0,1], [0,0,1,1]),
            # P3: U=[-1,1,0,0], V=[1,0,0,0], W=[0,0,1,0]
            ([-1,1,0,0], [1,0,0,0], [0,0,1,0]),
            # P4: U=[0,0,0,1], V=[0,1,0,-1], W=[1,0,0,-1]
            ([0,0,0,1], [0,1,0,-1], [1,0,0,-1]),
            # P5: U=[1,1,0,0], V=[1,0,0,0], W=[1,1,0,0]
            ([1,1,0,0], [1,0,0,0], [1,1,0,0]),
            # P6: U=[-1,0,1,0], V=[1,0,0,1], W=[-1,0,0,1]
            ([-1,0,1,0], [1,0,0,1], [-1,0,0,1]),
            # P7: U=[0,1,0,1], V=[0,1,0,-1], W=[1,0,0,0]
            ([0,1,0,1], [0,1,0,-1], [1,0,0,0]),
        ]
        
        for u_c, v_c, w_c in terms:
            for i in range(4):
                if u_c[i] != 0:
                    U[a_idx[i]][offset] += u_c[i]
                if v_c[i] != 0:
                    V[b_idx[i]][offset] += v_c[i]
                if w_c[i] != 0:
                    W[c_idx[i]][offset] += w_c[i]
            offset += 1
    
    def add_schoolbook_2x4x1(a_row_offset, a_col_offset, b_row_offset, b_col_offset, c_row_offset, c_col_offset):
        """Add schoolbook decomposition for 2x4 x 4x1 -> 2x1 block. Rank 8."""
        nonlocal offset
        # A is 2x4, B is 4x1, C is 2x1
        for i in range(2):
            for j in range(4):
                for k in range(1):
                    U[(a_row_offset + i) * m + (a_col_offset + j)][offset] = 1
                    V[(b_row_offset + j) * p + (b_col_offset + k)][offset] = 1
                    W[(c_row_offset + k) * n + (c_row_offset + i)][offset] = 1  # WRONG
                    # C index: c is 2x1, so C[i,0] -> flat index c_row_offset*n + c_col_offset + i
                    # Actually C is stored as p*n, so C[k*n+i] where k is column of C, i is row of C
                    # C is 2x1, so it has 2 rows and 1 column.
                    # In the W matrix (p*n), the index is k*n + i where k is the column index in B/C and i is the row index in A/C
                    # For C_tr (2x1): rows 0,1 of C correspond to i=0,1 in A_top; column 0 corresponds to k=4 in B
                    # W index = k*n + i = 4*3 + i = 12 + i
                    # But c_row_offset and c_col_offset are given in terms of the C block position.
                    # Let me reconsider the indexing.
                    offset += 1
    
    # Let me redo this more carefully with explicit index calculations.
    
    # Reset
    offset = 0
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: C_tl (2x2) = A_top(2x4) x B_left(4x2)
    # A_top: rows 0-1 of A, cols 0-3 of A
    # B_left: rows 0-3 of B, cols 0-1 of B
    # C_tl: rows 0-1 of C, cols 0-1 of C
    
    # Split A_top into A_tl(2x2) cols 0-1 and A_tr(2x2) cols 2-3
    # Split B_left into B_ll(2x2) rows 0-1 and B_lr(2x2) rows 2-3
    # C_tl = A_tl x B_ll + A_tr x B_lr
    
    # Strassen for A_tl(2x2) x B_ll(2x2) -> C_tl(2x2)
    add_strassen_2x2x2(0, 0, 0, 0, 0, 0)
    
    # Strassen for A_tr(2x2) x B_lr(2x2) -> C_tl(2x2)
    add_strassen_2x2x2(0, 2, 2, 0, 0, 0)
    
    # Block 2: C_tm (2x2) = A_top(2x4) x B_mid(4x2)
    # B_mid: rows 0-3 of B, cols 2-3 of B
    # C_tm: rows 0-1 of C, cols 2-3 of C
    
    # Split A_top into A_tl(2x2) cols 0-1 and A_tr(2x2) cols 2-3
    # Split B_mid into B_ml(2x2) rows 0-1 and B_mr(2x2) rows 2-3
    # C_tm = A_tl x B_ml + A_tr x B_mr
    
    add_strassen_2x2x2(0, 0, 0, 2, 0, 2)
    add_strassen_2x2x2(0, 2, 2, 2, 0, 2)
    
    # Block 3: C_tr (2x1) = A_top(2x4) x B_right(4x1)
    # B_right: rows 0-3 of B, col 4 of B
    # C_tr: rows 0-1 of C, col 4 of C
    # Schoolbook: rank 8
    for i in range(2):
        for j in range(4):
            U[i * m + j][offset] = 1
            V[j * p + 4][offset] = 1
            W[4 * n + i][offset] = 1
            offset += 1
    
    # Block 4: C_bl (1x2) = A_bot(1x4) x B_left(4x2)
    # A_bot: row 2 of A, cols 0-3
    # B_left: rows 0-3 of B, cols 0-1
    # C_bl: row 2 of C, cols 0-1
    # Schoolbook: rank 8
    for j in range(4):
        for k in range(2):
            U[2 * m + j][offset] = 1
            V[j * p + k][offset] = 1
            W[k * n + 2][offset] = 1
            offset += 1
    
    # Block 5: C_bm (1x2) = A_bot(1x4) x B_mid(4x2)
    # B_mid: rows 0-3 of B, cols 2-3
    # C_bm: row 2 of C, cols 2-3
    for j in range(4):
        for k in range(2):
            U[2 * m + j][offset] = 1
            V[j * p + (2 + k)][offset] = 1
            W[(2 + k) * n + 2][offset] = 1
            offset += 1
    
    # Block 6: C_br (1x1) = A_bot(1x4) x B_right(4x1)
    # Schoolbook: rank 4
    for j in range(4):
        U[2 * m + j][offset] = 1
        V[j * p + 4][offset] = 1
        W[4 * n + 2][offset] = 1
        offset += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

