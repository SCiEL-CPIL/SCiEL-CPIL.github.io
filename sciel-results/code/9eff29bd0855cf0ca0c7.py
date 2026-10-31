"""Strassen recursive 4x4x4 matrix multiplication tensor decomposition (rank 49)."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Recursive Strassen decomposition: rank = 7^2 = 49 for 4x4x4.
    # Each 2x2 block product uses Strassen's 7-multiplication formula.
    # W follows the official trace convention W[k*n + i, r].
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # Strassen 2x2x2: 7 multiplications
    # For C = A*B where A, B are 2x2:
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11) B00
    # M3 = A00 (B01 - B11)
    # M4 = A11 (B10 - B00)
    # M5 = (A00 + A01) B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Build U (16 x rank), V (16 x rank), W (16 x rank)
    # U row index: i*m + j (i in 0..3, j in 0..3)
    # V row index: j*p + k (j in 0..3, k in 0..3)
    # W row index: k*n + i (k in 0..3, i in 0..3)
    
    # For 4x4, split into 2x2 blocks:
    # Block indices: A[bi][bj] where bi,bj in {0,1}
    # A[bi][bj] corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # So element A[i][j] has block (i//2, j//2) and local pos (i%2, j%2)
    
    rank = 49
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(16)]
    W = [[0] * rank for _ in range(16)]
    
    # Strassen coefficients for 2x2x2
    # M1: A00+A11, B00+B11
    # M2: A10+A11, B00
    # M3: A00, B01-B11
    # M4: A11, B10-B00
    # M5: A00+A01, B11
    # M6: A10-A00, B00+B01
    # M7: A01-A11, B10+B11
    
    # For each block pair (bi, bj) of A and (bk, bj) of B... 
    # Wait, let me think more carefully.
    
    # A is 4x4, B is 4x4, C = A*B is 4x4.
    # Split A into 2x2 blocks A[bi][bj], B into B[bj][bk], C into C[bi][bk].
    # C[bi][bk] = sum_j A[bi][bj] * B[bj][bk]
    
    # For each block multiplication C[bi][bk] = sum_{bj=0,1} A[bi][bj] * B[bj][bk],
    # we apply Strassen within each 2x2 block product.
    
    # But Strassen gives us 7 multiplications for the 2x2 block product.
    # The "multiplications" in Strassen are products of linear combinations.
    
    # For block product A[bi][bj] * B[bj][bk] (each 2x2):
    # The 7 Strassen terms give contributions to C[bi][bk].
    
    # Let me define the Strassen terms for a 2x2x2:
    # Input A 2x2: A[0][0], A[0][1], A[1][0], A[1][1]
    # Input B 2x2: B[0][0], B[0][1], B[1][0], B[1][1]
    # Output C 2x2: C[0][0], C[0][1], C[1][0], C[1][1]
    
    # Strassen 7 terms:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Now for 4x4x4:
    # A[i][j] where i,j in 0..3. Block A[bi][bj] has elements A[2*bi+di][2*bj+dj] for di,dj in 0,1.
    # Similarly B[2*bj+dj][2*bk+dk] and C[2*bi+di][2*bk+dk].
    
    # For each (bi, bk) in {0,1}^2, C[bi][bk] = A[bi][0]*B[0][bk] + A[bi][1]*B[1][bk]
    # Each of these 2x2 block products uses Strassen's 7 terms.
    
    # So total rank = 4 block pairs * 7 Strassen terms = 28? No wait...
    
    # Actually, C[bi][bk] = sum_{bj} A[bi][bj] * B[bj][bk]
    # For fixed (bi, bk), we have 2 terms (bj=0 and bj=1), each a 2x2x2 Strassen.
    # That gives 2*7 = 14 per (bi,bk) pair, total 4*14 = 56. That's the naive block Strassen.
    
    # But the recursive approach: we can combine the two bj terms.
    # Actually no - the recursive Strassen for 4x4x4 works differently.
    
    # The proper recursive Strassen: treat 4x4 as 2x2 blocks of 2x2 matrices.
    # Then C = A*B where A, B, C are 2x2 blocks of 2x2 matrices.
    # Apply Strassen at the block level: 7 block multiplications.
    # Each block multiplication is a 2x2x2 matrix product, which itself uses Strassen: 7 scalar multiplications.
    # Total: 7 * 7 = 49.
    
    # So the structure is:
    # Level 1: 7 Strassen terms, each involving products of 2x2 block sums/differences.
    # Level 2: Each 2x2 block product uses 7 Strassen terms.
    
    # Let me define the block-level Strassen:
    # A blocks: A00, A01, A10, A11 (each 2x2)
    # B blocks: B00, B01, B10, B11 (each 2x2)
    # C blocks: C00, C01, C10, C11 (each 2x2)
    
    # S1 = (A00 + A11) * (B00 + B11)
    # S2 = (A10 + A11) * B00
    # S3 = A00 * (B01 - B11)
    # S4 = A11 * (B10 - B00)
    # S5 = (A00 + A01) * B11
    # S6 = (A10 - A00) * (B00 + B01)
    # S7 = (A01 - A11) * (B10 + B11)
    
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # Each S_t is a 2x2x2 matrix product (of 2x2 matrices), decomposed via Strassen into 7 scalar mults.
    
    # Total rank = 7 (block level) * 7 (within each block product) = 49.
    
    # Now let's build the factor matrices.
    
    # Strassen 2x2x2 coefficients:
    # For a 2x2x2 with A elements indexed (a,b), B elements (b,c), C elements (a,c):
    # U_A: row (a,b) -> coefficients for each of 7 terms
    # V_B: row (b,c) -> coefficients for each of 7 terms  
    # W_C: row (a,c) -> coefficients for each of 7 terms
    
    # Strassen term definitions (local 2x2 indices 0,1):
    # Term 1: A: (0,0)+1, (1,1)+1; B: (0,0)+1, (1,1)+1; C: (0,0)+1, (1,1)-1, (0,1)+1, (1,0)-1
    # Wait, let me be more careful.
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)(B00)
    # M3 = (A00)(B01-B11)
    # M4 = (A11)(B10-B00)
    # M5 = (A00+A01)(B11)
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # So for each term t (1-7), we have:
    # U_A[t][a][b]: coefficient of A[a][b] in the A-part of term t
    # V_B[t][b][c]: coefficient of B[b][c] in the B-part of term t
    # W_C[t][a][c]: coefficient in C[a][c] from term t
    
    # Term 1: A-part: A00=1, A11=1; B-part: B00=1, B11=1
    # Term 2: A-part: A10=1, A11=1; B-part: B00=1
    # Term 3: A-part: A00=1; B-part: B01=1, B11=-1
    # Term 4: A-part: A11=1; B-part: B10=1, B00=-1
    # Term 5: A-part: A00=1, A01=1; B-part: B11=1
    # Term 6: A-part: A10=1, A00=-1; B-part: B00=1, B01=1
    # Term 7: A-part: A01=1, A11=-1; B-part: B10=1, B11=1
    
    # C contributions:
    # C00: M1(+1), M4(+1), M5(-1), M7(+1)
    # C01: M3(+1), M5(+1)
    # C10: M2(+1), M4(+1)
    # C11: M1(+1), M2(-1), M3(+1), M6(+1)
    
    # Define local 2x2x2 Strassen factors (as dictionaries or arrays)
    # Local indices: a,b,c in {0,1}
    
    # U_local[t][a][b] for t in 0..6 (terms 1-7)
    U_local = [
        # Term 1
        [[1, 0], [0, 1]],
        # Term 2
        [[0, 0], [1, 1]],
        # Term 3
        [[1, 0], [0, 0]],
        # Term 4
        [[0, 0], [0, 1]],
        # Term 5
        [[1, 1], [0, 0]],
        # Term 6
        [[-1, 0], [1, 0]],
        # Term 7
        [[0, 1], [0, -1]],
    ]
    
    # V_local[t][b][c] for t in 0..6
    V_local = [
        # Term 1
        [[1, 0], [0, 1]],
        # Term 2
        [[1, 0], [0, 0]],
        # Term 3
        [[0, 1], [0, -1]],
        # Term 4
        [[-1, 0], [1, 0]],
        # Term 5
        [[0, 0], [0, 1]],
        # Term 6
        [[1, 1], [0, 0]],
        # Term 7
        [[0, 1], [1, 1]],
    ]
    
    # W_local[t][a][c] for t in 0..6
    W_local = [
        # Term 1: C00+=1, C11+=1
        [[1, 0], [0, 1]],
        # Term 2: C10+=1, C11+=-1
        [[0, 0], [1, -1]],
        # Term 3: C01+=1, C11+=1
        [[0, 1], [0, 1]],
        # Term 4: C00+=1, C10+=1
        [[1, 0], [1, 0]],
        # Term 5: C00+=-1, C01+=1
        [[-1, 1], [0, 0]],
        # Term 6: C11+=1
        [[0, 0], [0, 1]],
        # Term 7: C00+=1
        [[1, 0], [0, 0]],
    ]
    
    # Now build the full 4x4x4 decomposition.
    # Block-level Strassen: 7 terms, each is a product of 2x2 blocks.
    # A blocks: A[bi][bj] for bi,bj in {0,1}
    # B blocks: B[bj][bk] for bj,bk in {0,1}
    # C blocks: C[bi][bk] for bi,bk in {0,1}
    
    # Block-level Strassen terms:
    # S1 = (A00 + A11) * (B00 + B11)
    # S2 = (A10 + A11) * B00
    # S3 = A00 * (B01 - B11)
    # S4 = A11 * (B10 - B00)
    # S5 = (A00 + A01) * B11
    # S6 = (A10 - A00) * (B00 + B01)
    # S7 = (A01 - A11) * (B10 + B11)
    
    # For each block-level term s (0-6), we have:
    # A-block part: sum over (bi,bj) of coeff_s_A[bi][bj] * A[bi][bj]
    # B-block part: sum over (bj,bk) of coeff_s_B[bj][bk] * B[bj][bk]
    # C-block part: sum over (bi,bk) of coeff_s_C[bi][bk] * C[bi][bk]
    
    # Then each block product is further decomposed using local Strassen.
    
    # A-block coefficients for each block-level term:
    # S1: A00=1, A11=1
    # S2: A10=1, A11=1
    # S3: A00=1
    # S4: A11=1
    # S5: A00=1, A01=1
    # S6: A10=1, A00=-1
    # S7: A01=1, A11=-1
    
    # A_block_coeffs[s][bi][bj]
    A_block_coeffs = [
        # S1
        [[1, 0], [0, 1]],
        # S2
        [[0, 0], [1, 1]],
        # S3
        [[1, 0], [0, 0]],
        # S4
        [[0, 0], [0, 1]],
        # S5
        [[1, 1], [0, 0]],
        # S6
        [[-1, 0], [1, 0]],
        # S7
        [[0, 1], [0, -1]],
    ]
    
    # B-block coefficients for each block-level term:
    # S1: B00=1, B11=1
    # S2: B00=1
    # S3: B01=1, B11=-1
    # S4: B10=1, B00=-1
    # S5: B11=1
    # S6: B00=1, B01=1
    # S7: B10=1, B11=1
    
    # B_block_coeffs[s][bj][bk]
    B_block_coeffs = [
        # S1
        [[1, 0], [0, 1]],
        # S2
        [[1, 0], [0, 0]],
        # S3
        [[0, 1], [0, -1]],
        # S4
        [[-1, 0], [1, 0]],
        # S5
        [[0, 0], [0, 1]],
        # S6
        [[1, 1], [0, 0]],
        # S7
        [[0, 1], [1, 1]],
    ]
    
    # C-block coefficients for each block-level term:
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6
    
    # C_block_coeffs[s][bi][bk]
    C_block_coeffs = [
        # S1
        [[1, 0], [0, 1]],
        # S2
        [[0, 0], [1, -1]],
        # S3
        [[0, 1], [0, 1]],
        # S4
        [[1, 0], [1, 0]],
        # S5
        [[-1, 1], [0, 0]],
        # S6
        [[0, 0], [0, 1]],
        # S7
        [[1, 0], [0, 0]],
    ]
    
    # Now, for each block-level term s (0-6) and each local Strassen term t (0-6):
    # The scalar multiplication is:
    # (sum_{bi,bj,di,dj} A_block_coeffs[s][bi][bj] * U_local[t][di][dj] * A[2*bi+di][2*bj+dj])
    # *
    # (sum_{bj,bk,dj,dk} B_block_coeffs[s][bj][bk] * V_local[t][dj][dk] * B[2*bj+dj][2*bk+dk])
    #
    # And contributes to C:
    # sum_{bi,bk,di,dk} C_block_coeffs[s][bi][bk] * W_local[t][di][dk] * C[2*bi+di][2*bk+dk]
    
    # So for the global factor matrices:
    # U[2*bi+di, 2*bj+dj] (i.e., U[i][j] where i=2*bi+di, j=2*bj+dj)
    #   gets contribution A_block_coeffs[s][bi][bj] * U_local[t][di][dj] at column (s*7+t)
    #
    # V[2*bj+dj, 2*bk+dk] (i.e., V[j][k] where j=2*bj+dj, k=2*bk+dk)
    #   gets contribution B_block_coeffs[s][bj][bk] * V_local[t][dj][dk] at column (s*7+t)
    #
    # W[2*bk+dk, 2*bi+di] (i.e., W[k][i] where k=2*bk+dk, i=2*bi+di)
    #   gets contribution C_block_coeffs[s][bi][bk] * W_local[t][di][dk] at column (s*7+t)
    
    # Note: W index is k*n+i = k*4+i. So W[2*bk+dk][2*bi+di].
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            for bi in range(2):
                for bj in range(2):
                    for di in range(2):
                        for dj in range(2):
                            a_coeff = A_block_coeffs[s][bi][bj] * U_local[t][di][dj]
                            if a_coeff != 0:
                                i = 2 * bi + di
                                j = 2 * bj + dj
                                U[i][col] += a_coeff
            for bj in range(2):
                for bk in range(2):
                    for dj in range(2):
                        for dk in range(2):
                            b_coeff = B_block_coeffs[s][bj][bk] * V_local[t][dj][dk]
                            if b_coeff != 0:
                                j = 2 * bj + dj
                                k = 2 * bk + dk
                                V[j][col] += b_coeff
            for bi in range(2):
                for bk in range(2):
                    for di in range(2):
                        for dk in range(2):
                            c_coeff = C_block_coeffs[s][bi][bk] * W_local[t][di][dk]
                            if c_coeff != 0:
                                i = 2 * bi + di
                                k = 2 * bk + dk
                                W[k * n + i][col] += c_coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
