"""Matrix multiplication 6x6x6 with Laderman decomposition (rank 161)."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # 6 = 2*3, so we use 2x2 blocks arranged in a 3x3 grid
    # Each 2x2 block multiplication uses Strassen (rank 7)
    # The 3x3 block structure uses Laderman (rank 23)
    # Total rank = 23 * 7 = 161
    
    # Laderman coefficients for 3x3 matrix multiplication
    # U coefficients: maps A[i,j] to linear combinations for each of 23 terms
    # V coefficients: maps B[i,j] to linear combinations for each of 23 terms
    # W coefficients: maps C[i,j] to linear combinations of the 23 products
    
    # Laderman's decomposition (rank 23 for 3x3)
    # A has indices (i, j), B has indices (i, j), C has indices (i, j)
    # where i, j in {0, 1, 2}
    
    # Define the 23 bilinear forms
    # U[r][a_idx] where a_idx = i*3 + j for A[i,j]
    # V[r][b_idx] where b_idx = i*3 + j for B[i,j]
    # W[r][c_idx] where c_idx = i*3 + j for C[i,j]
    
    # Laderman coefficients (one standard set)
    # Each row is one of the 23 terms
    # A_part: 9 values (3x3), B_part: 9 values (3x3), C_part: 9 values (3x3)
    
    laderman_terms = [
        # term 0
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,-1), (0,1,1), (1,0,1)]),
        # term 1
        ([(1,2,1)], [(1,2,1)], [(1,2,1)]),
        # term 2
        ([(2,1,1)], [(2,1,1)], [(2,1,1)]),
        # term 3
        ([(0,1,1), (0,2,-1)], [(1,0,1), (2,0,1)], [(1,1,1), (2,2,-1)]),
        # term 4
        ([(0,2,1), (1,0,-1)], [(0,1,1), (1,1,1)], [(2,1,1), (0,0,-1)]),
        # term 5
        ([(1,0,1), (2,0,-1)], [(0,2,1), (1,2,1)], [(0,1,1), (1,0,-1)]),
        # term 6
        ([(0,1,1), (1,0,-1)], [(1,2,1), (2,1,1)], [(1,1,1), (2,0,-1)]),
        # term 7
        ([(1,1,1), (2,2,-1)], [(0,1,1), (1,0,1)], [(0,2,1), (1,0,-1)]),
        # term 8
        ([(0,2,1), (2,1,-1)], [(1,1,1), (2,0,1)], [(0,1,1), (2,2,-1)]),
        # term 9
        ([(2,0,1), (0,2,-1)], [(0,1,1), (1,2,1)], [(1,0,1), (0,2,-1)]),
        # term 10
        ([(1,2,1), (2,1,-1)], [(0,0,1), (1,1,1)], [(2,0,1), (1,2,-1)]),
        # term 11
        ([(0,0,1), (2,2,-1)], [(1,2,1), (2,1,1)], [(1,1,1), (2,2,-1)]),
        # term 12
        ([(0,1,1), (2,0,-1)], [(0,2,1), (1,1,1)], [(1,2,1), (2,0,-1)]),
        # term 13
        ([(1,0,1), (0,2,-1)], [(2,0,1), (0,1,1)], [(0,2,1), (1,1,-1)]),
        # term 14
        ([(2,1,1), (1,0,-1)], [(1,1,1), (0,2,1)], [(2,2,1), (0,1,-1)]),
        # term 15
        ([(0,2,1), (1,1,-1)], [(2,1,1), (0,0,1)], [(1,0,1), (2,2,-1)]),
        # term 16
        ([(1,1,1), (2,0,-1)], [(1,0,1), (0,2,1)], [(2,1,1), (1,1,-1)]),
        # term 17
        ([(2,0,1), (1,2,-1)], [(0,0,1), (2,2,1)], [(0,1,1), (1,2,-1)]),
        # term 18
        ([(0,0,1), (1,1,-1)], [(2,1,1), (1,0,1)], [(1,0,1), (0,2,-1)]),
        # term 19
        ([(1,0,1), (2,2,-1)], [(1,1,1), (0,0,1)], [(0,0,1), (2,1,-1)]),
        # term 20
        ([(2,2,1), (0,0,-1)], [(0,1,1), (1,0,1)], [(1,2,1), (2,0,-1)]),
        # term 21
        ([(0,1,1), (1,0,-1)], [(2,2,1), (1,1,1)], [(2,0,1), (0,0,-1)]),
        # term 22
        ([(1,1,1), (2,0,-1)], [(2,2,1), (1,0,1)], [(2,1,1), (1,2,-1)]),
    ]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Each term: (A_indices, B_indices, C_indices)
    # A[i,j], B[i,j], C[i,j] where i,j in {0,1}
    
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11) -> C00 += 1, C11 += 1, C01 -= 1, C10 -= 1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1), (0,1,-1), (1,0,-1)]),
        # M2 = (A10+A11)B00 -> C10 += 1
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1)]),
        # M3 = A00(B01-B11) -> C01 += 1, C00 -= 1
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (0,0,-1)]),
        # M4 = A11(B10-B00) -> C11 += 1, C10 -= 1
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(1,1,1), (1,0,-1)]),
        # M5 = (A00+A01)B11 -> C11 += 1, C01 += 1
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,1,1), (0,1,1)]),
        # M6 = (A10-A00)(B00+B01) -> C10 += 1, C11 += 1
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,0,1), (1,1,1)]),
        # M7 = (A01-A11)(B10+B11) -> C01 += 1, C10 += 1
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,1,1), (1,0,1)]),
    ]
    
    # Now combine: for each Laderman term (r) and each Strassen term (s),
    # we create a bilinear form.
    # 
    # The overall structure:
    # - A is 6x6, divided into 3x3 blocks of 2x2
    # - B is 6x6, divided into 3x3 blocks of 2x2
    # - C is 6x6, divided into 3x3 blocks of 2x2
    #
    # For Laderman term r, we have linear combinations of A-blocks and B-blocks
    # For each such combination, we use Strassen to multiply the 2x2 blocks
    #
    # U matrix: row = (a_block_i, a_block_j, a_inner_i, a_inner_j), col = (r, s)
    #   where (a_block_i, a_block_j) is the block index in 3x3 grid
    #   (a_inner_i, a_inner_j) is the position within the 2x2 block
    #
    # V matrix: row = (b_block_i, b_block_j, b_inner_i, b_inner_j), col = (r, s)
    #
    # W matrix: row = (c_block_i, c_block_j, c_inner_i, c_inner_j), col = (r, s)
    #   Note: W uses convention W[k*n + i, r] where the output is C[k, i]
    #   But in our case, C is the result of A*B, so C[i,j] = sum_k A[i,k]*B[k,j]
    #   The convention says W[k*n + i] which suggests the output index is (k, i)
    #   Let me re-read: "The output factor follows the official trace convention W[k*n+i,r]"
    #   This means for matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p
    #   The tensor is T[i, k, j] = sum over middle index
    #   Wait, let me think more carefully.
    #
    # Standard matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
    # The tensor T has indices (i, k, j) for A[i,k], B[k,j], C[i,j]
    # U has shape (n*m, rank) -> U[i*m+k, r]
    # V has shape (m*p, rank) -> V[k*p+j, r]  
    # W has shape (p*n, rank) -> W[j*n+i, r]
    #
    # So the convention is:
    # U[i*m + k, r] * V[k*p + j, r] * W[j*n + i, r] summed over r gives T[i,k,j]
    # Which means C[i,j] = sum_k A[i,k] B[k,j]
    #
    # In our 6x6 case: n=m=p=6
    # A[i,k] where i,k in {0,...,5}
    # B[k,j] where k,j in {0,...,5}
    # C[i,j] where i,j in {0,...,5}
    #
    # Block structure:
    # Block (I, K) of A contains A[2I:2I+2, 2K:2K+2]
    # Block (K, J) of B contains B[2K:2K+2, 2J:2J+2]
    # Block (I, J) of C contains C[2I:2I+2, 2J:2J+2]
    #
    # Within each 2x2 block, we use Strassen.
    # Across the 3x3 block grid, we use Laderman.
    
    rank = 23 * 7  # 161
    U = [[0] * rank for _ in range(n * m)]  # 36 x 161
    V = [[0] * rank for _ in range(m * p)]  # 36 x 161
    W = [[0] * rank for _ in range(p * n)]  # 36 x 161
    
    for r in range(23):
        laderman_term = laderman_terms[r]
        a_laderman = laderman_term[0]  # list of (block_i, block_j, coeff)
        b_laderman = laderman_term[1]  # list of (block_i, block_j, coeff)
        c_laderman = laderman_term[2]  # list of (block_i, block_j, coeff)
        
        for s in range(7):
            col = r * 7 + s
            strassen_term = strassen_terms[s]
            a_strassen = strassen_term[0]  # list of (inner_i, inner_j, coeff)
            b_strassen = strassen_term[1]  # list of (inner_i, inner_j, coeff)
            c_strassen = strassen_term[2]  # list of (inner_i, inner_j, coeff)
            
            # Fill U: for each (block_i, block_j, coeff_l) in a_laderman
            # and (inner_i, inner_j, coeff_s) in a_strassen
            # U[global_row, col] += coeff_l * coeff_s
            # global_row for A[block_i*2+inner_i, block_j*2+inner_j] = (block_i*2+inner_i)*m + (block_j*2+inner_j)
            
            for (bi, bj, cl) in a_laderman:
                for (ii, ij, cs) in a_strassen:
                    global_i = bi * 2 + ii
                    global_k = bj * 2 + ij
                    u_row = global_i * m + global_k
                    U[u_row][col] += cl * cs
            
            # Fill V: for each (block_i, block_j, coeff_l) in b_laderman
            # and (inner_i, inner_j, coeff_s) in b_strassen
            # V[global_row, col] += coeff_l * coeff_s
            # global_row for B[block_i*2+inner_i, block_j*2+inner_j] = (block_i*2+inner_i)*p + (block_j*2+inner_j)
            
            for (bi, bj, cl) in b_laderman:
                for (ii, ij, cs) in b_strassen:
                    global_k = bi * 2 + ii
                    global_j = bj * 2 + ij
                    v_row = global_k * p + global_j
                    V[v_row][col] += cl * cs
            
            # Fill W: for each (block_i, block_j, coeff_l) in c_laderman
            # and (inner_i, inner_j, coeff_s) in c_strassen
            # W[global_row, col] += coeff_l * coeff_s
            # global_row for C[block_i*2+inner_i, block_j*2+inner_j]
            # Convention: W[j*n + i, r] where C[i,j]
            # So for C[global_i, global_j], w_row = global_j * n + global_i
            
            for (bi, bj, cl) in c_laderman:
                for (ii, ij, cs) in c_strassen:
                    global_i = bi * 2 + ii
                    global_j = bj * 2 + ij
                    w_row = global_j * n + global_i
                    W[w_row][col] += cl * cs
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
