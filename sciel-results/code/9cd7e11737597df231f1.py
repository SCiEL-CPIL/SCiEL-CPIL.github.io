"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    import time
    start_time = time.time()
    
    # Strassen's 2x2 matrix multiplication: rank 7
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # For each Strassen term t (0..6), define:
    #   u_coeffs[t]: list of (i, j, coeff) for U factor (row i*m+j, sign)
    #   v_coeffs[t]: list of (j, k, coeff) for V factor (row j*p+k, sign)
    #   w_coeffs[t]: list of (i, j, coeff) for W factor (col i, row j, sign) for output C[i,j]
    
    # U factors (coefficients for A submatrix):
    # t=0: A00+A11 -> rows (0,0) and (1,1) with coeff +1
    # t=1: A10+A11 -> rows (1,0) and (1,1) with coeff +1
    # t=2: A00 -> row (0,0) with coeff +1
    # t=3: A11 -> row (1,1) with coeff +1
    # t=4: A00+A01 -> rows (0,0) and (0,1) with coeff +1
    # t=5: A10-A00 -> rows (1,0) with +1, (0,0) with -1
    # t=6: A01-A11 -> rows (0,1) with +1, (1,1) with -1
    U_strassen = [
        [(0, 0, 1), (1, 1, 1)],   # t=0
        [(1, 0, 1), (1, 1, 1)],   # t=1
        [(0, 0, 1)],              # t=2
        [(1, 1, 1)],              # t=3
        [(0, 0, 1), (0, 1, 1)],   # t=4
        [(1, 0, 1), (0, 0, -1)],  # t=5
        [(0, 1, 1), (1, 1, -1)],  # t=6
    ]
    
    # V factors (coefficients for B submatrix):
    # t=0: B00+B11 -> cols (0,0) and (1,1) with coeff +1
    # t=1: B00 -> col (0,0) with coeff +1
    # t=2: B01-B11 -> cols (0,1) with +1, (1,1) with -1
    # t=3: B10-B00 -> cols (1,0) with +1, (0,0) with -1
    # t=4: B11 -> col (1,1) with coeff +1
    # t=5: B00+B01 -> cols (0,0) and (0,1) with coeff +1
    # t=6: B10+B11 -> cols (1,0) and (1,1) with coeff +1
    V_strassen = [
        [(0, 0, 1), (1, 1, 1)],   # t=0
        [(0, 0, 1)],              # t=1
        [(0, 1, 1), (1, 1, -1)],  # t=2
        [(1, 0, 1), (0, 0, -1)],  # t=3
        [(1, 1, 1)],              # t=4
        [(0, 0, 1), (0, 1, 1)],   # t=5
        [(1, 0, 1), (1, 1, 1)],   # t=6
    ]
    
    # W factors (coefficients for C output):
    # C00 = M1+M4-M5+M7 -> for term t, W gets contribution to C[0,0]
    # C01 = M3+M5 -> contribution to C[0,1]
    # C10 = M2+M4 -> contribution to C[1,0]
    # C11 = M1-M2+M3+M6 -> contribution to C[1,1]
    # 
    # For each t, which C elements get contributions?
    # t=0 (M1): C00(+1), C11(+1)
    # t=1 (M2): C10(+1), C11(-1)
    # t=2 (M3): C01(+1), C11(+1)
    # t=3 (M4): C00(+1), C10(+1)
    # t=4 (M5): C00(-1), C01(+1)
    # t=5 (M6): C11(+1)
    # t=6 (M7): C00(+1)
    W_strassen = [
        [(0, 0, 1), (1, 1, 1)],   # t=0: C00+1, C11+1
        [(1, 0, 1), (1, 1, -1)],  # t=1: C10+1, C11-1
        [(0, 1, 1), (1, 1, 1)],   # t=2: C01+1, C11+1
        [(0, 0, 1), (1, 0, 1)],   # t=3: C00+1, C10+1
        [(0, 0, -1), (0, 1, 1)],  # t=4: C00-1, C01+1
        [(1, 1, 1)],              # t=5: C11+1
        [(0, 0, 1)],              # t=6: C00+1
    ]
    
    n, m, p = DIMENSIONS  # 4, 4, 4
    
    # For 4x4 matrix mult, we use 2x2 blocks of size 2x2.
    # The 4x4 matrix A is partitioned into 2x2 blocks:
    # A[0:2, 0:2], A[0:2, 2:4], A[2:4, 0:2], A[2:4, 2:4]
    # Block (bi, bj) corresponds to A rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # 
    # For each Strassen term t (outer, block level) and s (inner, entry level),
    # we create one rank factor. Total rank = 7 * 7 = 49.
    #
    # U[row, factor]: row = 2*(block_i) + row_in_block, col within block
    # For 4x4: U has 16 rows (4*4), each factor has 49 columns
    # U[i*4+j, c] where i,j in 0..3
    #
    # The block-level Strassen gives us which 2x2 block of A to use (with sign)
    # The entry-level Strassen gives us which element within that 2x2 block
    #
    # Actually, the Kronecker product approach:
    # Rank 49 decomposition: for each pair (t_block, t_entry) where t in 0..6
    # U factor: for block-level t, pick block (bi, bj); for entry-level s, pick position (ri, cj) within block
    # So U row index = (2*bi + ri)*4 + (2*bj + cj)
    
    # Entry-level Strassen for 2x2 scalar "matrices" (just picking elements):
    # This is the same as the block-level structure but for 2x2 entries
    # For a 2x2 matrix, the "Strassen" for picking a single element is trivial:
    # We just pick A[ri, cj] for specific ri, cj
    #
    # Wait, I need to think about this more carefully.
    # 
    # The 4x4 matrix multiplication tensor T[i,j,k] = delta[i,j] * delta[j,k]
    # where C[i,k] = sum_j A[i,j] * B[j,k]
    #
    # Using 2x2 blocks: C_block[bi,bk] = sum_bj A_block[bi,bj] * B_block[bj,bk]
    # Strassen decomposes this block multiplication into 7 terms.
    # 
    # Then within each block, we have 2x2 matrix multiplication, which is just
    # C[bi*2+ri, bk*2+cj] = sum_rj A[bi*2+ri, bj*2+rj] * B[bj*2+rj, bk*2+cj]
    # This inner 2x2 mult can also be done with Strassen (rank 7) or schoolbook (rank 4).
    # 
    # Using Strassen for both levels: rank = 7 * 7 = 49
    # Using Strassen for block, schoolbook for inner: rank = 7 * 4 = 28 (but need to check validity)
    # Actually, we can use different decompositions at each level.
    # 
    # Let me use Strassen at block level (7 terms) and schoolbook at entry level (4 terms per block pair)
    # That gives rank = 7 * 4 = 28. But wait, that might not be correct because the inner
    # multiplication is not independent across block pairs.
    #
    # Actually, the cleanest approach: the 4x4 matrix mult tensor is the Kronecker product
    # of the 2x2 matrix mult tensor with itself. If T_2 is the 2x2 matrix mult tensor
    # (rank 7 via Strassen), then T_4 = T_2 ⊗ T_2 has rank at most 7*7 = 49.
    #
    # For the Kronecker product decomposition:
    # T_4[i1*i2 + i2', j1*j2 + j2', k1*k2 + k2'] 
    #   = sum over (t1, t2) U_2[i1*i2+i2', t1*t2] * V_2[j1*j2+j2', t1*t2] * W_2[k1*k2+k2', t1*t2]
    #
    # Wait, that's not quite right. Let me think again.
    #
    # The 4x4 matrix A can be seen as a 2x2 block matrix where each block is 2x2.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2.
    # C = A * B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #              [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    #
    # Strassen for the block multiplication:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # Each Mi is a 2x2 matrix product. For each Mi, we need to decompose the 2x2
    # matrix multiplication. Using schoolbook: rank 4 per Mi, total 7*4 = 28.
    # Using Strassen for inner: rank 7 per Mi, total 7*7 = 49.
    #
    # Let's use schoolbook for the inner 2x2 multiplication to get rank 28.
    #
    # For each Strassen block term t (0..6) and inner schoolbook term s (0..3):
    # The inner schoolbook for 2x2: C[ri, cj] = sum_rj A[ri, rj] * B[rj, cj]
    # Terms: (ri=0,rj=0,cj=0), (ri=0,rj=1,cj=1), (ri=1,rj=0,cj=0), (ri=1,rj=1,cj=1)
    
    # Define inner schoolbook terms: (row_in_block, col_in_block_for_A, row_in_block_for_B, col_in_block_for_B)
    # For C[ri, cj] = sum_rj A[ri,rj]*B[rj,cj], schoolbook has 4 terms:
    # s=0: ri=0, rj=0, cj=0
    # s=1: ri=0, rj=1, cj=1  -- wait, that's not right
    # Schoolbook: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    #            C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    #            C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    #            C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1]
    # So 8 terms total for schoolbook 2x2. But we can group them.
    # Actually schoolbook 2x2 has rank 4 if we use the fact that it's just 4 independent products?
    # No, 2x2 matrix mult has 8 multiplications in schoolbook, but the tensor rank is 4? 
    # No, the tensor rank of 2x2 matrix multiplication is 7 (Strassen).
    # Schoolbook gives rank 8 (or 4 if you count differently).
    # 
    # Let me just use the Kronecker product of two Strassen decompositions: rank 49.
    # This is clean and well-understood.
    
    rank = 49
    U = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
    V = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows
    W = [[0] * rank for _ in range(16)]  # 4*4 = 16 rows, W[k*4+i, c]
    
    # For Kronecker product: index in 4x4 = (block_index, entry_index)
    # A[i,j] where i = 2*bi + ri, j = 2*bj + cj
    # U row index = i*4 + j = (2*bi+ri)*4 + (2*bj+cj)
    # V row index = j*4 + k = (2*bj+cj)*4 + (2*bk+ck)
    # W col index: W[k*4 + i, c] = (2*bk+ck)*4 + (2*bi+ri)
    
    # For each pair (t, s) where t is block-level Strassen term, s is entry-level Strassen term:
    # factor_index = t * 7 + s
    
    # Block-level Strassen coefficients:
    # For term t, U gets contributions from specific blocks (bi, bj) with coefficients
    # U_strassen[t] = list of (bi, bj, coeff)
    
    # Entry-level Strassen: same structure but for 2x2 entries within a block
    # For a 2x2 "matrix" (really just 4 scalars), Strassen picks specific elements
    # This is the same as the block-level coefficients applied to the 2x2 grid
    
    # Actually, for the Kronecker product, both levels use the same Strassen structure.
    # At the entry level, we're doing "2x2 matrix multiplication" where the "matrices" 
    # are 1x1 (scalars), but structured as 2x2. The Strassen decomposition applies identically.
    
    for t in range(7):
        for s in range(7):
            col = t * 7 + s
            # U contributions from block level and entry level
            for (bi, bj, cb) in U_strassen[t]:
                for (ri, cj, ce) in U_strassen[s]:
                    row = (2 * bi + ri) * 4 + (2 * bj + cj)
                    U[row][col] += cb * ce
            
            # V contributions
            for (bj, bk, cb) in V_strassen[t]:
                for (cj, ck, ce) in V_strassen[s]:
                    row = (2 * bj + cj) * 4 + (2 * bk + ck)
                    V[row][col] += cb * ce
            
            # W contributions: W[k*4+i, col]
            # W_strassen[t] gives (ci, ck, coeff) for block-level C
            # W_strassen[s] gives (ri, rj, coeff) for entry-level C
            for (ci, ck, cb) in W_strassen[t]:
                for (ri, rj, ce) in W_strassen[s]:
                    i = 2 * ci + ri
                    k = 2 * ck + rj
                    W[k * 4 + i][col] += cb * ce
    
    # Verify: convert to numpy and check
    import numpy as np
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    # Check: for random A, B, does sum_c U[:,c] * V[:,c] * W[:,c] give correct C?
    # The decomposition means: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    # where T[i,j,k] = 1 if i==j (for A[i,j]*B[j,k]) ... 
    # Actually T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k]
    # So T[i,j,k] = delta[i,k]... no.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor is T[i,j,k] = 1 for all i,j,k? No.
    # T[i,j,k] is the coefficient such that C[i,k] = sum_{j} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if the term A[i,j]*B[j,k] appears in C[i,k], which is always when indices match.
    # T[i,j,k] = 1 for all valid i,j,k where the product contributes to C[i,k].
    # Actually T is defined as: the matrix mult tensor has entries T[i,j,k] = delta_{i,k} ... no.
    
    # Let me just verify numerically.
    np.random.seed(42)
    A = np.random.randint(-2, 3, size=(4, 4)).astype(float)
    B = np.random.randint(-2, 3, size=(4, 4)).astype(float)
    C_expected = A @ B
    
    # Compute C from decomposition:
    # C[i,k] = sum_j sum_c U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]
    C_computed = np.zeros((4, 4))
    for i in range(4):
        for k in range(4):
            total = 0
            for j in range(4):
                for c in range(rank):
                    total += U[i * 4 + j][c] * V[j * 4 + k][c] * W[k * 4 + i][c]
            C_computed[i, k] = total
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook if verification fails
        rank = 64
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        column = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U[i * 4 + j][column] = 1
                    V[j * 4 + k][column] = 1
                    W[k * 4 + i][column] = 1
                    column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
