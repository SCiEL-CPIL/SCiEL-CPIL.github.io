import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    # 4x4x4 via Strassen: 7 block-mults × 7 rank-1 = 49
    # 4x4x3 via 4 blocks of 2x2x3: 4 × 11 = 44
    total_rank = 49 + 44

    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]

    current_rank = 0

    # --- Part 1: 4x4x4 (columns 0..3 of B) ---
    # Strassen 2x2x2 rank-7 terms: (u4, v4, w4)
    base_2x2x2 = [
        ([1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]),  # (A11+A22)(B11+B22)
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # (A21+A22)B11
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # A11(B21-B11)
        ([0, 0, 0, 1], [0, -1, 1, 0], [0, 1, 1, 0]),   # A22(B12-B22)
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # (A11+A12)B22
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),   # (A21-A11)(B11+B12)
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # (A12-A22)(B21+B22)
    ]

    # 7 Strassen multipliers for 4x4x4: (x_def, y_def, c_def)
    # x_def: {block_idx: coeff} for A blocks, y_def for B blocks, c_def for C blocks
    strassen_4x4 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),      # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),            # M2
        ({0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1}),            # M3
        ({3: 1}, {1: 1, 3: -1}, {0: 1, 2: 1}),            # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),            # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),            # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),            # M7
    ]

    # Block layout: A[i,j] with i=block_row*2+local_i, j=block_col*2+local_j
    # Block b: row=b//2, col=b%2
    def a_idx(b, li):
        return (b // 2) * 2 + li  # row offset in A
    def a_col(b, lj):
        return (b % 2) * 2 + lj   # col offset in A

    def b_row(b, li):
        return (b // 2) * 2 + li  # row offset in B
    def b_col(b, lj):
        return (b % 2) * 2 + lj   # col offset in B

    def c_row(b, li):
        return (b // 2) * 2 + li  # row offset in C
    def c_col(b, lj):
        return (b % 2) * 2 + lj   # col offset in C

    for s in range(7):
        x_def, y_def, c_def = strassen_4x4[s]
        for t in range(7):
            u4, v4, w4 = base_2x2x2[t]
            r = current_rank
            current_rank += 1

            # Fill U: for each A-block in x_def, for each local (li,lj)
            for b_a, ca in x_def.items():
                for li in range(2):
                    for lj in range(2):
                        u_val = u4[li * 2 + lj] * ca
                        if u_val != 0:
                            row = a_idx(b_a, li) * m + a_col(b_a, lj)
                            U[row][r] += u_val

            # Fill V: for each B-block in y_def
            for b_b, cb in y_def.items():
                for li in range(2):
                    for lj in range(2):
                        v_val = v4[li * 2 + lj] * cb
                        if v_val != 0:
                            row = b_row(b_b, li) * p + b_col(b_b, lj)
                            V[row][r] += v_val

            # Fill W: for each C-block in c_def
            for b_c, cc in c_def.items():
                for li in range(2):
                    for lj in range(2):
                        w_val = w4[li * 2 + lj] * cc
                        if w_val != 0:
                            row = c_row(b_c, li) * n + c_col(b_c, lj)
                            W[row][r] += w_val

    # --- Part 2: 4x4x3 (columns 4..6 of B) ---
    # Split into 4 blocks: 2x2x2 (cols 4-5) and 2x2x1 (col 6)
    # 2x2x2 rank 7, 2x2x1 rank 4 (schoolbook: 2x2x1 = 4 terms)
    # Total per block-pair: 7 + 4 = 11, times 2 row-blocks = 22... 
    # Actually: 4 row-blocks of A (2x2 each) × 2 col-groups of B (2x2 and 2x1)
    # Each 2x2x2: rank 7, each 2x2x1: rank 4
    # 4 blocks × (7+4) = 44

    # 2x2x3 rank-11 decomposition:
    # For 2x2x2 part (cols 0,1 of the 3): use Strassen rank 7
    # For 2x2x1 part (col 2): schoolbook rank 4
    base_2x2x3 = []
    # 2x2x2 Strassen terms (u4, v2 for B[0,0],B[1,0], w2 for C[0,0],C[1,0])
    # Actually B is 2x3, C is 2x2... let me think differently.
    # A is 2x2, B is 2x3, C is 2x2
    # T[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,1} (first two cols)
    # Plus T[i,2] = sum_j A[i,j]*B[j,2]
    
    # For the 2x2x2 part (first 2 cols of B):
    # Use Strassen with u4 (A), v4 (B first 2 cols as 2x2), w4 (C)
    # For the 2x2x1 part (3rd col of B):
    # Schoolbook: 4 terms, each A[i,j]*B[j,2] -> C[i,2]
    
    # Let me define 2x2x3 as: rank-11 = 7 (for 2x2x2) + 4 (for 2x2x1)
    # u4 = A entries, v6 = B entries (2x3=6), w4 = C entries (2x2=4)
    
    base_2x2x3 = []
    # 7 Strassen terms for the 2x2x2 part (B cols 0,1)
    for t in range(7):
        u4, v4, w4 = base_2x2x2[t]
        # v4 corresponds to B[0,0],B[0,1],B[1,0],B[1,1] -> indices 0,1,2,3 in v6
        v6 = [0.0] * 6
        v6[0] = v4[0]  # B[0,0]
        v6[1] = v4[1]  # B[0,1]
        v6[2] = v4[2]  # B[1,0]
        v6[3] = v4[3]  # B[1,1]
        # v6[4], v6[5] = 0 (B[0,2], B[1,2])
        base_2x2x3.append((u4, v6, w4))
    
    # 4 schoolbook terms for the 2x2x1 part (B col 2)
    # C[i,2] = A[i,0]*B[0,2] + A[i,1]*B[1,2]
    # Term: A[0,0]*B[0,2] -> C[0,2]
    base_2x2x3.append(([1,0,0,0], [0,0,0,0,1,0], [0,0,1,0]))
    # Term: A[0,1]*B[1,2] -> C[0,2]
    base_2x2x3.append(([0,1,0,0], [0,0,0,0,0,1], [0,0,1,0]))
    # Term: A[1,0]*B[0,2] -> C[1,2]
    base_2x2x3.append(([0,0,1,0], [0,0,0,0,1,0], [0,0,0,1]))
    # Term: A[1,1]*B[1,2] -> C[1,2]
    base_2x2x3.append(([0,0,0,1], [0,0,0,0,0,1], [0,0,0,1]))

    # Now apply to 4 blocks of A (2x2 each) for B cols 4-6
    # A block b: rows 2*(b//2) to 2*(b//2)+1, cols 2*(b%2) to 2*(b%2)+1
    # B: rows 0-1 (for each A block row), cols 4-6
    # C: rows 0-1 (for each A block row), cols 0-1 (for each A block col)
    
    for b_a in range(4):
        a_row_off = (b_a // 2) * 2
        a_col_off = (b_a % 2) * 2
        c_row_off = (b_a // 2) * 2
        c_col_off = (b_a % 2) * 2
        
        for t in range(11):
            u4, v6, w4 = base_2x2x3[t]
            r = current_rank
            current_rank += 1
            
            # U: A entries
            for li in range(2):
                for lj in range(2):
                    u_val = u4[li * 2 + lj]
                    if u_val != 0:
                        row = (a_row_off + li) * m + (a_col_off + lj)
                        U[row][r] += u_val
            
            # V: B entries (2 rows × 3 cols = 6 entries)
            # B row = a_row_off + li (same row block as A), col = 4 + lj
            for li in range(2):
                for lj in range(3):
                    v_val = v6[li * 3 + lj]
                    if v_val != 0:
                        row = (a_row_off + li) * p + (4 + lj)
                        V[row][r] += v_val
            
            # W: C entries (2 rows × 2 cols = 4 entries)
            for li in range(2):
                for lj in range(2):
                    w_val = w4[li * 2 + lj]
                    if w_val != 0:
                        row = (c_row_off + li) * n + (c_col_off + lj)
                        W[row][r] += w_val

    # Trim to actual rank used
    U = [row[:current_rank] for row in U]
    V = [row[:current_rank] for row in V]
    W = [row[:current_rank] for row in W]

    return {"U": U, "V": V, "W": W}
