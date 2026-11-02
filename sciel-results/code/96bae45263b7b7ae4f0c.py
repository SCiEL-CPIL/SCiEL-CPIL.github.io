"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    # Verify function
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Construct rank 81 solution: 49 (for 4x4x4) + 32 (for 4x4x2)
    # 4x4x4 uses recursive Strassen (rank 49)
    # 4x4x2 uses schoolbook (rank 32)
    
    # Strassen coefficients for 2x2 matrix multiplication
    A_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                  # M3
        {(1,1): 1},                  # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    B_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                  # M2
        {(0,1): 1, (1,1): -1},       # M3
        {(1,0): 1, (0,0): -1},       # M4
        {(1,1): 1},                  # M5
        {(0,0): 1, (0,1): 1},        # M6
        {(1,0): 1, (1,1): 1}         # M7
    ]
    C_coeffs = [
        [
            [1, 0, 0, -1, 1, 0, -1],  # C00
            [0, 0, 1, 0, 1, 0, 0]     # C01
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0]    # C11
        ]
    ]
    
    # Block-level coefficients for outer Strassen
    block_A_coeffs = [
        [[1, 0], [0, 1]],    # M1: A00 + A11
        [[0, 0], [1, 1]],    # M2: A10 + A11
        [[1, 0], [0, 0]],    # M3: A00
        [[0, 0], [0, 1]],    # M4: A11
        [[1, 1], [0, 0]],    # M5: A00 + A01
        [[0, 0], [1, -1]],   # M6: A10 - A00
        [[0, 1], [0, -1]]    # M7: A01 - A11
    ]
    block_B_coeffs = [
        [[1, 0], [0, 1]],    # M1: B00 + B11
        [[1, 0], [0, 0]],    # M2: B00
        [[0, 1], [0, -1]],   # M3: B01 - B11
        [[0, -1], [1, 0]],   # M4: B10 - B00
        [[0, 0], [0, 1]],    # M5: B11
        [[1, 1], [0, 0]],    # M6: B00 + B01
        [[0, 1], [1, 1]]     # M7: B10 + B11
    ]
    
    # Build 4x4x4 decomposition using recursive Strassen (rank 49)
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    for beta in range(7):
        for alpha in range(7):
            col = beta * 7 + alpha
            
            # Build U_444 for this term
            for i in range(4):
                for j in range(4):
                    block_i = i // 2
                    block_j = j // 2
                    
                    coeff = 0
                    for r in range(2):
                        for s in range(2):
                            if (r, s) in A_coeffs[alpha]:
                                a_coeff = A_coeffs[alpha][(r, s)]
                                b_coeff = block_A_coeffs[beta][r][s]
                                if r == block_i and s == block_j:
                                    coeff += a_coeff * b_coeff
                    U_444[i*4+j][col] = coeff
            
            # Build V_444 for this term
            for j in range(4):
                for k in range(4):
                    block_j = j // 2
                    block_k = k // 2
                    
                    coeff = 0
                    for r in range(2):
                        for s in range(2):
                            if (r, s) in B_coeffs[alpha]:
                                b_coeff = B_coeffs[alpha][(r, s)]
                                a_coeff = block_B_coeffs[beta][r][s]
                                if r == block_j and s == block_k:
                                    coeff += b_coeff * a_coeff
                    V_444[j*4+k][col] = coeff
            
            # Build W_444 for this term
            for bi in range(2):
                for bk in range(2):
                    for p2 in range(2):
                        for q in range(2):
                            c_i = 2*bi + p2
                            c_k = 2*bk + q
                            coeff = C_coeffs[bi][bk][beta] * C_coeffs[p2][q][alpha]
                            W_444[c_k*4+c_i][col] += coeff
    
    # Build 4x4x2 part using schoolbook (rank 32)
    rank_442 = 4 * 4 * 2
    U_442 = [[0]*rank_442 for _ in range(16)]
    V_442 = [[0]*rank_442 for _ in range(8)]
    W_442 = [[0]*rank_442 for _ in range(8)]
    
    col_442 = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_442[i*4+j][col_442] = 1
                V_442[j*2+k][col_442] = 1
                W_442[k*4+i][col_442] = 1
                col_442 += 1
    
    # Combine: total rank = 49 + 32 = 81
    total_rank = 49 + 32
    U_total = [[0]*total_rank for _ in range(16)]
    V_total = [[0]*total_rank for _ in range(24)]
    W_total = [[0]*total_rank for _ in range(24)]
    
    # Copy 4x4x4 part (columns 0-48)
    for i in range(16):
        for c in range(49):
            U_total[i][c] = U_444[i][c]
            V_total[i][c] = V_444[i][c]
            W_total[i][c] = W_444[i][c]
    
    # Copy 4x4x2 part (columns 49-80)
    for i in range(16):
        for c in range(32):
            U_total[i][49+c] = U_442[i][c]
    for j in range(8):
        for c in range(32):
            V_total[j][49+c] = V_442[j][c]
    for k in range(8):
        for c in range(32):
            W_total[k][49+c] = W_442[k][c]
    
    # Verify the combined solution
    if verify(U_total, V_total, W_total, total_rank, 4, 4, 6):
        best = {'U': U_total, 'V': V_total, 'W': W_total, 'rank': total_rank}
    
    # Now try to improve by using a better decomposition for the 4x4x2 part
    # or by using a different splitting strategy
    
    # Alternative: split into 2x4x6 and 2x4x6 (partition first index)
    # Or: use Strassen on 2x2 blocks for the 4x4x2 part as well
    
    # Let's try: split the 4x4x2 into two 2x4x2 parts (partition first index)
    # Each 2x4x2 can be done with rank 2*4*2 = 16 schoolbook, or potentially less
    
    # Actually, let's try a different approach: use Strassen-like for 2x2 blocks
    # on the 4x4x2 part
    
    # For 4x4x2: treat as 2x2 block matrix (2x2 blocks of 2x2) times 2x1 block matrix (2x1 blocks of 2x2)
    # Result is 2x1 block matrix (2x1 blocks of 2x2)
    
    # A = [A00 A01; A10 A11] (2x2 blocks)
    # B = [B00; B10] (2x1 blocks, each 2x2)
    # C = [A00*B00 + A01*B10; A10*B00 + A11*B10]
    
    # We can use 4 multiplications: A00*B00, A01*B10, A10*B00, A11*B10
    # Each is 2x2 x 2x2 -> 2x2, which has rank 7 (Strassen)
    # Total rank: 4 * 7 = 28
    
    # But we can do better with Strassen-like:
    # M1 = (A00 + A11) * B00  -- wait, B has only one column of blocks
    # Actually for 2x2 * 2x1, we can use:
    # M1 = (A00 + A10) * B00
    # M2 = A10 * B10  
    # M3 = A00 * B10
    # M4 = (A11 + A01) * B00  -- hmm, this doesn't work cleanly
    
    # Let me just use the 4 schoolbook multiplications, each of rank 7:
    # Total rank for 4x4x2: 4 * 7 = 28
    
    # Actually, let's be more careful. For 2x2 x 2x2 -> 2x2, rank is 7.
    # For 4x4x2 = (2x2 blocks) * (2x1 blocks of 2x2), we have 4 block multiplications.
    # If each uses rank 7, total is 28.
    
    # Let's build this:
    rank_442_strassen = 28
    U_442s = [[0]*rank_442_strassen for _ in range(16)]
    V_442s = [[0]*rank_442_strassen for _ in range(8)]
    W_442s = [[0]*rank_442_strassen for _ in range(8)]
    
    # Block multiplications:
    # C00 = A00*B00 + A01*B10  (top-left 2x2 of C)
    # C10 = A10*B00 + A11*B10  (bottom-left 2x2 of C)
    # Wait, C is 4x2, so C[i,k] for i in 0..3, k in 0..1
    # C00 = C[0:2, 0:2], C10 = C[2:4, 0:2]
    
    # A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
    # A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # B00 = B[0:2, 0:2], B10 = B[2:4, 0:2]
    
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    
    # 4 Strassen decompositions:
    # For A00*B00: use Strassen, contributes to C00
    # For A01*B10: use Strassen, contributes to C00
    # For A10*B00: use Strassen, contributes to C10
    # For A11*B10: use Strassen, contributes to C10
    
    # Each Strassen has 7 terms.
    
    # Strassen for 2x2*2x2->2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each block multiplication, we need to map the Strassen coefficients
    # to the appropriate positions in U, V, W.
    
    # Let's define helper to add a Strassen decomposition for a 2x2*2x2->2x2 product
    # at specific positions in A, B, C.
    
    def add_strassen_2x2(U_mat, V_mat, W_mat, rank_offset, 
                         a_row_off, a_col_off,  # top-left of A block in A
                         b_row_off, b_col_off,  # top-left of B block in B
                         c_row_off, c_col_off,  # top-left of C block in C
                         scale=1):
        """Add Strassen decomposition for A_block * B_block -> C_block."""
        # A_block is 2x2 at (a_row_off, a_col_off) in A
        # B_block is 2x2 at (b_row_off, b_col_off) in B
        # C_block is 2x2 at (c_row_off, c_col_off) in C
        
        # Strassen coefficients (same as before but for local 2x2 indices)
        # Local indices: 0,1 for rows/cols of each 2x2 block
        
        A_loc = [
            {(0,0): 1, (1,1): 1},       # M1
            {(1,0): 1, (1,1): 1},       # M2
            {(0,0): 1},                  # M3
            {(1,1): 1},                  # M4
            {(0,0): 1, (0,1): 1},       # M5
            {(1,0): 1, (0,0): -1},      # M6
            {(0,1): 1, (1,1): -1}       # M7
        ]
        B_loc = [
            {(0,0): 1, (1,1): 1},       # M1
            {(0,0): 1},                  # M2
            {(0,1): 1, (1,1): -1},       # M3
            {(1,0): 1, (0,0): -1},       # M4
            {(1,1): 1},                  # M5
            {(0,0): 1, (0,1): 1},        # M6
            {(1,0): 1, (1,1): 1}         # M7
        ]
        C_loc = [
            [
                [1, 0, 0, -1, 1, 0, -1],  # C00
                [0, 0, 1, 0, 1, 0, 0]     # C01
            ],
            [
                [0, 1, 0, 1, 0, 0, 0],    # C10
                [1, -1, 1, 0, 0, 1, 0]    # C11
            ]
        ]
        
        for mu in range(7):
            col = rank_offset + mu
            # U: A_block contributions
            for r in range(2):
                for s in range(2):
                    if (r, s) in A_loc[mu]:
                        row = a_row_off + r
                        col_a = a_col_off + s
                        U_mat[row][col] += scale * A_loc[mu][(r, s)]
            
            # V: B_block contributions
            for r in range(2):
                for s in range(2):
                    if (r, s) in B_loc[mu]:
                        row = b_row_off + r
                        col_b = b_col_off + s
                        V_mat[row][col] += scale * B_loc[mu][(r, s)]
            
            # W: C_block contributions
            for r in range(2):
                for s in range(2):
                    coeff = C_loc[r][s][mu]
                    if coeff != 0:
                        row = c_row_off + r
                        col_c = c_col_off + s
                        W_mat[row][col] += scale * coeff
    
    # For 4x4x2:
    # A is 4x4, B is 4x2, C is 4x2
    # A00 at (0,0), A01 at (0,2), A10 at (2,0), A11 at (2,2)
    # B00 at (0,0), B10 at (2,0)
    # C00 at (0,0), C10 at (2,0)
    
    # Four Strassen decompositions:
    # 1. A00 * B00 -> C00 (contributes to C[0:2, 0:2])
    add_strassen_2x2(U_442s, V_442s, W_442s, 0,
                     0, 0,  # A00
                     0, 0,  # B00
                     0, 0)  # C00
    
    # 2. A01 * B10 -> C00 (contributes to C[0:2, 0:2])
    add_strassen_2x2(U_442s, V_442s, W_442s, 7,
                     0, 2,  # A01
                     2, 0,  # B10
                     0, 0)  # C00
    
    # 3. A10 * B00 -> C10 (contributes to C[2:4, 0:2])
    add_strassen_2x2(U_442s, V_442s, W_442s, 14,
                     2, 0,  # A10
                     0, 0,  # B00
                     2, 0)  # C10
    
    # 4. A11 * B10 -> C10 (contributes to C[2:4, 0:2])
    add_strassen_2x2(U_442s, V_442s, W_442s, 21,
                     2, 2,  # A11
                     2, 0,  # B10
                     2, 0)  # C10
    
    # Verify the 4x4x2 Strassen part
    def verify_442(U, V, W, rank):
        for i in range(4):
            for j in range(4):
                for k in range(2):
                    s = 0
                    for c in range(rank):
                        s += U[i*4+j][c] * V[j*2+k][c] * W[k*4+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    if verify_442(U_442s, V_442s, W_442s, rank_442_strassen):
        # Combine with 4x4x4 part
        total_rank2 = 49 + rank_442_strassen
        U_total2 = [[0]*total_rank2 for _ in range(16)]
        V_total2 = [[0]*total_rank2 for _ in range(24)]
        W_total2 = [[0]*total_rank2 for _ in range(24)]
        
        for i in range(16):
            for c in range(49):
                U_total2[i][c] = U_444[i][c]
                V_total2[i][c] = V_444[i][c]
                W_total2[i][c] = W_444[i][c]
        
        for i in range(16):
            for c in range(rank_442_strassen):
                U_total2[i][49+c] = U_442s[i][c]
        for j in range(8):
            for c in range(rank_442_strassen):
                V_total2[j][49+c] = V_442s[j][c]
        for k in range(8):
            for c in range(rank_442_strassen):
                W_total2[k][49+c] = W_442s[k][c]
        
        if verify(U_total2, V_total2, W_total2, total_rank2, 4, 4, 6):
            best = {'U': U_total2, 'V': V_total2, 'W': W_total2, 'rank': total_rank2}
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
