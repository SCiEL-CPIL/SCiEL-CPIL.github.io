"""Matrix multiplication 4x4x8 rank minimization."""
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 4x4 x 4x8 matrix multiplication using Strassen's algorithm.
    
    We split the 4x8 matrix into two 4x4 halves: B = [B1 | B2]
    Then A*B = A*B1 | A*B2
    
    Each A*B_i (4x4 x 4x4) can be done with Strassen at rank 7.
    But we need to be careful: Strassen's 7-term algorithm gives rank 7 for 4x4x4.
    So total rank = 7 + 7 = 14? No wait, that's not right.
    
    Actually, let me reconsider. Strassen's algorithm for 4x4x4 matrix multiplication
    has rank 7. So for 4x4x8, if we split the output into two 4x4 blocks,
    each requiring rank 7, total rank = 14.
    
    But wait - Strassen's rank-7 decomposition is for the full 4x4x4 tensor.
    Let me verify: the standard Strassen decomposition for 2x2x2 has rank 7.
    For 4x4x4, we can use the 2x2x2 Strassen recursively, giving rank 7^2 = 49.
    
    Actually no. The 4x4x4 matrix multiplication tensor can be decomposed with rank 49
    using the standard block Strassen (treating 4x4 as 2x2 blocks of 2x2).
    But there might be better decompositions.
    
    Actually, for 4x4x4, the minimal rank is known to be at most 49 (from Strassen recursion)
    but could be lower. The best known upper bound for 4x4x4 is 49.
    
    So for 4x4x8: rank = 49 + 49 = 98 (splitting into two 4x4x4 problems).
    
    Let me implement the 49-rank Strassen decomposition for 4x4x4, then concatenate
    two copies for the 4x4x8 case.
    
    Strassen for 2x2x2:
    Given matrices A (2x2), B (2x2), compute C = A*B
    Split into quadrants:
    A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    
    M1 = (a00+a11)*(b00+b11)
    M2 = (a10+a11)*b00
    M3 = a00*(b01-b11)
    M4 = a11*(b10-b00)
    M5 = (a00+a01)*b11
    M6 = (a10-a00)*(b00+b10)
    M7 = (a01-a11)*(b01+b11)
    
    c00 = M1 + M4 - M5 + M7
    c01 = M3 + M5
    c10 = M2 + M4
    c11 = M1 - M2 + M3 + M6
    
    For 4x4x4, we treat the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    Each block is 2x2, and we apply Strassen to the block structure.
    This gives 7 recursive calls, each of which is a 2x2x2 Strassen (rank 7).
    Total rank = 7 * 7 = 49.
    
    Let me build this carefully.
    
    For 4x4x4:
    A is 4x4, B is 4x4, C = A*B is 4x4.
    
    Split A into 4 blocks of size 2x2: A00, A01, A10, A11
    Split B into 4 blocks of size 2x2: B00, B01, B10, B11
    
    C = [[C00, C01], [C10, C11]] where each Cij is 2x2
    
    Using Strassen at the block level:
    S1 = (A00 + A11) * (B00 + B11)
    S2 = (A10 + A11) * B00
    S3 = A00 * (B01 - B11)
    S4 = A11 * (B10 - B00)
    S5 = (A00 + A01) * B11
    S6 = (A10 - A00) * (B00 + B10)
    S7 = (A01 - A11) * (B01 + B11)
    
    C00 = S1 + S4 - S5 + S7
    C01 = S3 + S5
    C10 = S2 + S4
    C11 = S1 - S2 + S3 + S6
    
    Each S_i is a 2x2 matrix product, which we decompose using 2x2x2 Strassen (rank 7).
    
    So for each S_i, we have 7 rank-1 terms. Total: 7*7 = 49 rank-1 terms.
    
    Now let me index the elements:
    A[i,j] for i,j in 0..3
    B[j,k] for j,k in 0..3
    C[i,k] for i,k in 0..3
    
    Block indices:
    A block (bi, bj) contains A[2*bi + di, 2*bj + dj] for di,dj in 0,1
    B block (bj, bk) contains B[2*bj + dj, 2*bk + dk] for dj,dk in 0,1
    C block (bi, bk) contains C[2*bi + di, 2*bk + dk] for di,dk in 0,1
    
    For the 4x4x4 decomposition:
    U49: shape (16, 49) where row = i*4 + j (i=0..3, j=0..3)
    V49: shape (16, 49) where row = j*4 + k (j=0..3, k=0..3)
    W49: shape (16, 49) where row = k*4 + i (k=0..3, i=0..3)
    
    For each of the 7 block-level Strassen terms (t=1..7), and each of the 7
    2x2-level Strassen terms (s=1..7), we have a rank-1 term.
    
    Let me define the 2x2 Strassen coefficients.
    
    For 2x2x2 Strassen:
    Input: A (2x2), B (2x2), Output: C = A*B (2x2)
    
    The 7 rank-1 decompositions:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B01 - B11)
    M4 = A11 * (B10 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B10)
    M7 = (A01 - A11) * (B01 + B11)
    
    C00 = M1 - M5 + M7 + M4
    C01 = M5 + M3
    C10 = M2 + M4
    C11 = M6 + M3 - M2 + M1
    
    For each Mi, it's a product of a linear combination of A entries and a linear
    combination of B entries.
    
    Let me define:
    For 2x2 Strassen, the U coefficients (for A), V coefficients (for B), and 
    W coefficients (for C) for each of the 7 terms:
    
    Term 1: U: A00+ A11, V: B00+ B11, W: contributes +1 to C00, -1 to C11
    Term 2: U: A10+ A11, V: B00,       W: contributes +1 to C10, -1 to C11
    Term 3: U: A00,       V: B01- B11,  W: contributes +1 to C01, +1 to C11
    Term 4: U: A11,       V: B10- B00,  W: contributes +1 to C00, +1 to C10
    Term 5: U: A00+ A01, V: B11,       W: contributes -1 to C00, +1 to C01
    Term 6: U: A10- A00, V: B00+ B10,  W: contributes +1 to C11
    Term 7: U: A01- A11, V: B01+ B11,  W: contributes +1 to C00
    
    Let me write this more carefully.
    
    For 2x2 Strassen, with A indices (r,c) where r,c in {0,1}, B indices (c,d), C indices (r,d):
    
    U_coeffs[s][r,c] for s in 0..6:
    s=0: [[1,0],[1,0]]  (A00 + A11)
    s=1: [[0,0],[1,1]]  (A10 + A11)
    s=2: [[1,0],[0,0]]  (A00)
    s=3: [[0,0],[0,1]]  (A11)
    s=4: [[1,1],[0,0]]  (A00 + A01)
    s=5: [[-1,0],[1,0]] (A10 - A00)
    s=6: [[0,-1],[0,1]] (A01 - A11)
    
    V_coeffs[s][c,d] for s in 0..6:
    s=0: [[1,0],[1,0]]  (B00 + B11) -- wait, B is indexed (c,d) where c is row, d is col
    Actually B[c,d]:
    s=0: B00 + B11 = [[1,0],[0,1]]
    s=1: B00 = [[1,0],[0,0]]
    s=2: B01 - B11 = [[0,1],[0,-1]]
    s=3: B10 - B00 = [[-1,0],[1,0]]
    s=4: B11 = [[0,0],[0,1]]
    s=5: B00 + B10 = [[1,0],[1,0]]
    s=6: B01 + B11 = [[0,1],[0,1]]
    
    W_coeffs[s][r,d] for s in 0..6:
    C[r,d]:
    s=0: C00 += 1, C11 -= 1 => [[1,0],[0,-1]]
    s=1: C10 += 1, C11 -= 1 => [[0,0],[1,-1]]
    s=2: C01 += 1, C11 += 1 => [[0,1],[0,1]]
    s=3: C00 += 1, C10 += 1 => [[1,0],[1,0]]
    s=4: C00 -= 1, C01 += 1 => [[-1,1],[0,0]]
    s=5: C11 += 1 => [[0,0],[0,1]]
    s=6: C00 += 1 => [[1,0],[0,0]]
    
    Now for 4x4x4:
    We have 7 block-level terms t in 0..6, each corresponding to one of the 7 Strassen
    block products. For each block product, we need to identify which 2x2 blocks of A,
    B are involved and with what coefficients.
    
    Block-level Strassen:
    t=0: (A00+A11)*(B00+B11)
    t=1: (A10+A11)*B00
    t=2: A00*(B01-B11)
    t=3: A11*(B10-B00)
    t=4: (A00+A01)*B11
    t=5: (A10-A00)*(B00+B10)
    t=6: (A01-A11)*(B01+B11)
    
    For each t, the A blocks involved and their coefficients:
    t=0: A00(+1), A11(+1)
    t=1: A10(+1), A11(+1)
    t=2: A00(+1)
    t=3: A11(+1)
    t=4: A00(+1), A01(+1)
    t=5: A10(+1), A00(-1)
    t=6: A01(+1), A11(-1)
    
    For each t, the B blocks involved and their coefficients:
    t=0: B00(+1), B11(+1)
    t=1: B00(+1)
    t=2: B01(+1), B11(-1)
    t=3: B10(+1), B00(-1)
    t=4: B11(+1)
    t=5: B00(+1), B10(+1)
    t=6: B01(+1), B11(+1)
    
    For each (t, s) pair where s is the 2x2 Strassen term within the block product,
    we have a rank-1 contribution.
    
    The U vector for this term: for each (i,j) in 0..3 x 0..3,
    U[i,j] = sum over A-blocks (bi,bj) involved in t: coeff_A[t][bi,bj] * U_2x2[s][2*bi+di, 2*bj+dj]... 
    
    Wait, I need to be more careful. The A block (bi,bj) contains elements A[2*bi+di, 2*bj+dj]
    for di,dj in 0,1. The 2x2 Strassen U coefficient for element (di,dj) within the block is
    U_2x2[s][di,dj].
    
    So U[i,j] where i=2*bi+di, j=2*bj+dj:
    U[i,j] = sum over (bi',bj') in blocks involved in t: coeff_A[t][bi',bj'] * (U_2x2[s][di,dj] if (bi',bj')=(bi,bj) else 0)
    
    Actually, since each block-level term t involves specific A blocks, and within each
    block the 2x2 Strassen applies, the contribution to U[i,j] is:
    
    For i=2*bi+di, j=2*bj+dj:
    U[i,j] = coeff_A[t][bi,bj] * U_2x2[s][di,dj] if (bi,bj) is one of the A blocks in term t, else 0.
    
    Similarly for V and W.
    
    Let me code this up.
    """
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # 2x2 Strassen coefficients
    # U_2x2[s][r][c] for s in 0..6, r,c in 0,1
    U2 = [
        [[1,0],[1,0]],   # s=0: A00+A11
        [[0,0],[1,1]],   # s=1: A10+A11
        [[1,0],[0,0]],   # s=2: A00
        [[0,0],[0,1]],   # s=3: A11
        [[1,1],[0,0]],   # s=4: A00+A01
        [[-1,0],[1,0]],  # s=5: A10-A00
        [[0,-1],[0,1]],  # s=6: A01-A11
    ]
    
    # V_2x2[s][r][c] for s in 0..6, r,c in 0,1 (B indexed by row,col)
    V2 = [
        [[1,0],[0,1]],   # s=0: B00+B11
        [[1,0],[0,0]],   # s=1: B00
        [[0,1],[0,-1]],  # s=2: B01-B11
        [[-1,0],[1,0]],  # s=3: B10-B00
        [[0,0],[0,1]],   # s=4: B11
        [[1,0],[1,0]],   # s=5: B00+B10
        [[0,1],[0,1]],   # s=6: B01+B11
    ]
    
    # W_2x2[s][r][c] for s in 0..6, r,c in 0,1 (C indexed by row,col)
    W2 = [
        [[1,0],[0,-1]],  # s=0: C00+1, C11-1
        [[0,0],[1,-1]],  # s=1: C10+1, C11-1
        [[0,1],[0,1]],   # s=2: C01+1, C11+1
        [[1,0],[1,0]],   # s=3: C00+1, C10+1
        [[-1,1],[0,0]],  # s=4: C00-1, C01+1
        [[0,0],[0,1]],   # s=5: C11+1
        [[1,0],[0,0]],   # s=6: C00+1
    ]
    
    # Block-level Strassen: for each term t in 0..6, which A blocks and B blocks are involved
    # A_blocks[t] = list of (bi, bj, coeff)
    A_blocks = [
        [(0,0,1), (1,1,1)],    # t=0: A00+A11
        [(1,0,1), (1,1,1)],    # t=1: A10+A11
        [(0,0,1)],             # t=2: A00
        [(1,1,1)],             # t=3: A11
        [(0,0,1), (0,1,1)],    # t=4: A00+A01
        [(1,0,1), (0,0,-1)],   # t=5: A10-A00
        [(0,1,1), (1,1,-1)],   # t=6: A01-A11
    ]
    
    # B_blocks[t] = list of (bj, bk, coeff) where B block is (row_block, col_block)
    B_blocks = [
        [(0,0,1), (1,1,1)],    # t=0: B00+B11
        [(0,0,1)],             # t=1: B00
        [(0,1,1), (1,1,-1)],   # t=2: B01-B11
        [(1,0,1), (0,0,-1)],   # t=3: B10-B00
        [(1,1,1)],             # t=4: B11
        [(0,0,1), (1,0,1)],    # t=5: B00+B10
        [(0,1,1), (1,1,1)],    # t=6: B01+B11
    ]
    
    # For 4x4x4:
    # U49: shape (16, 49), row index = i*4+j
    # V49: shape (16, 49), row index = j*4+k
    # W49: shape (16, 49), row index = k*4+i
    
    rank_49 = 49
    U49 = [[0]*rank_49 for _ in range(16)]
    V49 = [[0]*rank_49 for _ in range(16)]
    W49 = [[0]*rank_49 for _ in range(16)]
    
    col = 0
    for t in range(7):
        for s in range(7):
            # This is rank-1 term at column col
            # U contributions:
            for (bi, bj, coeff_a) in A_blocks[t]:
                for di in range(2):
                    for dj in range(2):
                        i = 2*bi + di
                        j = 2*bj + dj
                        U49[i*4+j][col] += coeff_a * U2[s][di][dj]
            
            # V contributions:
            for (bj, bk, coeff_b) in B_blocks[t]:
                for dj in range(2):
                    for dk in range(2):
                        j = 2*bj + dj
                        k = 2*bk + dk
                        V49[j*4+k][col] += coeff_b * V2[s][dj][dk]
            
            # W contributions:
            # C block (bi, bk) where the Strassen term contributes
            # For term t, the C blocks involved:
            # t=0: C00(+1), C11(-1) -> from W2[s]
            # t=1: C10(+1), C11(-1)
            # t=2: C01(+1), C11(+1)
            # t=3: C00(+1), C10(+1)
            # t=4: C00(-1), C01(+1)
            # t=5: C11(+1)
            # t=6: C00(+1)
            
            # Actually, the W contributions depend on which C blocks the block-level
            # Strassen term contributes to, and then within each C block, the 2x2 Strassen
            # determines the local contribution.
            
            # Let me define C_blocks[t] = list of (bi, bk, coeff_c)
            C_blocks = [
                [(0,0,1), (1,1,-1)],   # t=0
                [(1,0,1), (1,1,-1)],   # t=1
                [(0,1,1), (1,1,1)],    # t=2
                [(0,0,1), (1,0,1)],    # t=3
                [(0,0,-1), (0,1,1)],   # t=4
                [(1,1,1)],             # t=5
                [(0,0,1)],             # t=6
            ]
            
            for (bi, bk, coeff_c) in C_blocks[t]:
                for di in range(2):
                    for dk in range(2):
                        i = 2*bi + di
                        k = 2*bk + dk
                        W49[k*4+i][col] += coeff_c * W2[s][di][dk]
            
            col += 1
    
    # Now for 4x4x8:
    # Split B into B1 (cols 0-3) and B2 (cols 4-7)
    # C = A*B1 | A*B2
    # Total rank = 49 + 49 = 98
    
    rank_total = 98
    U_total = [[0]*rank_total for _ in range(16)]  # n*m = 4*4 = 16
    V_total = [[0]*rank_total for _ in range(32)]  # m*p = 4*8 = 32
    W_total = [[0]*rank_total for _ in range(32)]  # p*n = 8*4 = 32
    
    # First 49 columns: A*B1
    for i in range(16):
        for c in range(49):
            U_total[i][c] = U49[i][c]
    
    for j in range(4):
        for k in range(4):
            # B1 has cols 0-3, so B1[j,k] corresponds to V49 row j*4+k
            v49_row = j*4 + k
            v_total_row = j*4 + k  # since B1 is at cols 0-3 of B
            for c in range(49):
                V_total[v_total_row][c] = V49[v49_row][c]
    
    for k in range(4):
        for i in range(4):
            # C1 has rows 0-3 (k=0..3), so C1[k,i] corresponds to W49 row k*4+i
            w49_row = k*4 + i
            w_total_row = k*4 + i  # since C1 is at rows 0-3 of C
            for c in range(49):
                W_total[w_total_row][c] = W49[w49_row][c]
    
    # Next 49 columns (49-97): A*B2
    for i in range(16):
        for c in range(49):
            U_total[i][49+c] = U49[i][c]
    
    for j in range(4):
        for k in range(4):
            # B2 has cols 4-7, so B2[j,k] = B[j, k+4]
            v49_row = j*4 + k
            v_total_row = j*4 + (k+4)  # since B2 is at cols 4-7 of B
            for c in range(49):
                V_total[v_total_row][49+c] = V49[v49_row][c]
    
    for k in range(4):
        for i in range(4):
            # C2 has rows 4-7 (k=4..7), so C2[k,i] = C[k+4,i]
            w49_row = k*4 + i
            w_total_row = (k+4)*4 + i  # since C2 is at rows 4-7 of C
            for c in range(49):
                W_total[w_total_row][49+c] = W49[w49_row][c]
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Should not happen for this static computation
        pass
    
    return {"U": U_total, "V": V_total, "W": W_total}
# EVOLVE-BLOCK-END
