def construct():
    n, m, p = 5, 5, 5
    
    # We will build the factors block by block.
    # Block (0,0): 3x3 A, 3x3 B, 3x3 C. Indices i,j,k in [0,2]
    # Block (0,1): 3x2 A, 2x2 B? No.
    # A is 5x5. Rows i, Cols j.
    # B is 5x5. Rows j, Cols k.
    # C is 5x5. Rows k, Cols i (Trace convention).
    
    # Split i, j, k into [0,2] and [3,4].
    # Let S0 = [0,1,2], S1 = [3,4].
    
    # Block A00: i in S0, j in S0. (3x3)
    # Block A01: i in S0, j in S1. (3x2)
    # Block A10: i in S1, j in S0. (2x3)
    # Block A11: i in S1, j in S1. (2x2)
    
    # Block B00: j in S0, k in S0. (3x3)
    # Block B01: j in S0, k in S1. (3x2)
    # Block B10: j in S1, k in S0. (2x3)
    # Block B11: j in S1, k in S1. (2x2)
    
    # Result C blocks:
    # C00: k in S0, i in S0. (3x3) <- From A00*B00, A01*B10
    # C01: k in S0, i in S1. (3x2) <- From A10*B00, A11*B10? No.
    # C_{ki} = sum_j A_{ij} B_{jk}.
    
    # C00 (k in S0, i in S0):
    # j in S0: A_{i,j} (i in S0, j in S0) * B_{j,k} (j in S0, k in S0) -> A00 * B00
    # j in S1: A_{i,j} (i in S0, j in S1) * B_{j,k} (j in S1, k in S0) -> A01 * B10
    
    # C01 (k in S0, i in S1):
    # j in S0: A_{i,j} (i in S1, j in S0) * B_{j,k} (j in S0, k in S0) -> A10 * B00
    # j in S1: A_{i,j} (i in S1, j in S1) * B_{j,k} (j in S1, k in S0) -> A11 * B10
    
    # C10 (k in S1, i in S0):
    # j in S0: A_{i,j} (i in S0, j in S0) * B_{j,k} (j in S0, k in S1) -> A00 * B01
    # j in S1: A_{i,j} (i in S0, j in S1) * B_{j,k} (j in S1, k in S1) -> A01 * B11
    
    # C11 (k in S1, i in S1):
    # j in S0: A_{i,j} (i in S1, j in S0) * B_{j,k} (j in S0, k in S1) -> A10 * B01
    # j in S1: A_{i,j} (i in S1, j in S1) * B_{j,k} (j in S1, k in S1) -> A11 * B11

    # We need decompositions for:
    # 1. 3x3 * 3x3 (A00, B00) -> C00_part1. Rank 27 (Strassen)
    # 2. 3x2 * 2x3 (A01, B10) -> C00_part2. Rank 18 (Trivial)
    # 3. 2x3 * 3x3 (A10, B00) -> C01_part1. Rank 18 (Trivial)
    # 4. 2x2 * 2x3 (A11, B10) -> C01_part2. Rank 12 (Trivial)
    # 5. 3x3 * 3x2 (A00, B01) -> C10_part1. Rank 18 (Trivial)
    # 6. 3x2 * 2x2 (A01, B11) -> C10_part2. Rank 12 (Trivial)
    # 7. 2x3 * 3x2 (A10, B01) -> C11_part1. Rank 12 (Trivial)
    # 8. 2x2 * 2x2 (A11, B11) -> C11_part2. Rank 7 (Strassen)

    # Total Rank: 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7 = 124.
    # This is only slightly better than 125.
    # Why? Because the cross terms are trivial.
    # Can we do better on cross terms?
    # 3x2 * 2x3 is 18.
    # 2x3 * 3x3 is 18.
    # It seems hard to beat 125 significantly with this split if cross terms are trivial.
    
    # However, 124 < 125. It is an improvement.
    # I will implement this.
    
    rank = 124
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    def add_block(U_block, V_block, W_block, i_offset, j_offset, k_offset, i_size, j_size, k_size, start_rank):
        # i_offset, j_offset, k_offset are the starting indices for i, j, k
        # i_size, j_size, k_size are the dimensions of the block
        # U_block has shape (i_size * j_size, rank_block)
        # V_block has shape (j_size * k_size, rank_block)
        # W_block has shape (k_size * i_size, rank_block)
        
        r_block = U_block[0].__len__()
        
        for r in range(r_block):
            for ii in range(i_size):
                for jj in range(j_size):
                    U[(i_offset + ii) * 5 + (j_offset + jj)][start_rank + r] += U_block[ii * j_size + jj][r]
                    
            for jj in range(j_size):
                for kk in range(k_size):
                    V[(j_offset + jj) * 5 + (k_offset + kk)][start_rank + r] += V_block[jj * k_size + kk][r]
                    
            for kk in range(k_size):
                for ii in range(i_size):
                    W[(k_offset + kk) * 5 + (i_offset + ii)][start_rank + r] += W_block[kk * i_size + ii][r]

    def trivial_decomp(i_size, j_size, k_size):
        # Returns U, V, W for trivial decomposition of i_size x j_size * j_size x k_size
        rank = i_size * j_size * k_size
        U = [[0] * rank for _ in range(i_size * j_size)]
        V = [[0] * rank for _ in range(j_size * k_size)]
        W = [[0] * rank for _ in range(k_size * i_size)]
        
        col = 0
        for i in range(i_size):
            for j in range(j_size):
                for k in range(k_size):
                    U[i * j_size + j][col] = 1
                    V[j * k_size + k][col] = 1
                    W[k * i_size + i][col] = 1
                    col += 1
        return U, V, W

    def strassen_2x2():
        # Returns U, V, W for 2x2 Strassen
        # U, V, W shape 4x7
        U = [[0] * 7 for _ in range(4)]
        V = [[0] * 7 for _ in range(4)]
        W = [[0] * 7 for _ in range(4)]
        
        # M1 = (a11+a22)(b11+b22)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1 # c11 += M1, c22 += M1
        
        # M2 = (a21+a22)b11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1 # c21 += M2, c22 -= M2
        
        # M3 = a11(b12-b22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1 # c12 += M3, c22 += M3
        
        # M4 = a22(b21-b11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1 # c11 += M4, c21 += M4
        
        # M5 = (a11+a12)b22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1 # c11 -= M5, c12 += M5
        
        # M6 = (a21-a11)(b11+b12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1 # c22 += M6
        
        # M7 = (a12-a22)(b21+b22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1 # c11 += M7
        
        return U, V, W

    def strassen_3x3():
        # Placeholder: Use trivial for now to ensure correctness, or implement Strassen 3x3?
        # Strassen 3x3 is complex. I will use Trivial 3x3 (Rank 27) to be safe.
        # Actually, trivial 3x3 is 27.
        return trivial_decomp(3, 3, 3)

    current_rank = 0
    
    # 1. A00 (3x3) * B00 (3x3) -> C00 (3x3)
    U_b, V_b, W_b = strassen_3x3()
    add_block(U_b, V_b, W_b, 0, 0, 0, 3, 3, 3, current_rank)
    current_rank += 27
    
    # 2. A01 (3x2) * B10 (2x3) -> C00 (3x3)
    U_b, V_b, W_b = trivial_decomp(3, 2, 3)
    add_block(U_b, V_b, W_b, 0, 3, 0, 3, 2, 3, current_rank)
    current_rank += 18
    
    # 3. A10 (2x3) * B00 (3x3) -> C01 (3x2)
    # i in S1 (2), j in S0 (3), k in S0 (3)
    # Output C is k (3) x i (2)
    U_b, V_b, W_b = trivial_decomp(2, 3, 3)
    add_block(U_b, V_b, W_b, 3, 0, 0, 2, 3, 3, current_rank)
    current_rank += 18
    
    # 4. A11 (2x2) * B10 (2x3) -> C01 (3x2)
    # i in S1 (2), j in S1 (2), k in S0 (3)
    # Output C is k (3) x i (2)
    U_b, V_b, W_b = trivial_decomp(2, 2, 3)
    add_block(U_b, V_b, W_b, 3, 3, 0, 2, 2, 3, current_rank)
    current_rank += 12
    
    # 5. A00 (3x3) * B01 (3x2) -> C10 (2x3)
    # i in S0 (3), j in S0 (3), k in S1 (2)
    # Output C is k (2) x i (3)
    U_b, V_b, W_b = trivial_decomp(3, 3, 2)
    add_block(U_b, V_b, W_b, 0, 0, 3, 3, 3, 2, current_rank)
    current_rank += 18
    
    # 6. A01 (3x2) * B11 (2x2) -> C10 (2x3)
    # i in S0 (3), j in S1 (2), k in S1 (2)
    # Output C is k (2) x i (3)
    U_b, V_b, W_b = trivial_decomp(3, 2, 2)
    add_block(U_b, V_b, W_b, 0, 3, 3, 3, 2, 2, current_rank)
    current_rank += 12
    
    # 7. A10 (2x3) * B01 (3x2) -> C11 (2x2)
    # i in S1 (2), j in S0 (3), k in S1 (2)
    # Output C is k (2) x i (2)
    U_b, V_b, W_b = trivial_decomp(2, 3, 2)
    add_block(U_b, V_b, W_b, 3, 0, 3, 2, 3, 2, current_rank)
    current_rank += 12
    
    # 8. A11 (2x2) * B11 (2x2) -> C11 (2x2)
    # i in S1 (2), j in S1 (2), k in S1 (2)
    # Output C is k (2) x i (2)
    U_b, V_b, W_b = strassen_2x2()
    add_block(U_b, V_b, W_b, 3, 3, 3, 2, 2, 2, current_rank)
    current_rank += 7
    
    return {"U": U, "V": V, "W": W}
