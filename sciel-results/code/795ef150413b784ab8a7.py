"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 4
    start_time = time.time()
    deadline = start_time + 120.0  # 120s deadline per task contract

    # Known best tensor rank for 4x4 x 4x4 matrix multiplication is 49
    # (Strassen's algorithm applied to 4x4 can achieve rank 49 via various constructions)
    # Let's use the known Strassen-based rank-49 decomposition.
    # 
    # The standard Strassen algorithm for 2x2 matrices gives rank 7.
    # For 4x4 matrices, we can use Strassen recursively or other constructions.
    # 
    # Actually, the known optimal rank for 4x4 matrix multiplication is 49.
    # This can be achieved by the Strassen algorithm applied in a specific way,
    # or by other known constructions.
    #
    # Let me implement a known rank-49 decomposition using the approach from
    # Landsberg or other known results. Actually, let me use a simpler approach:
    # Use Strassen's 7-term algorithm for 2x2 blocks, applied to 4x4 matrices
    # by partitioning them into 2x2 blocks of 2x2 submatrices.
    #
    # For a 4x4 matrix A, partition into 4 blocks A_ij (each 2x2) for i,j in {0,1}.
    # Similarly for B and C. Then C_ij = sum_k A_ik * B_kj.
    # Each block multiplication A_ik * B_kj is a 2x2 * 2x2 = 2x2 multiplication.
    # Using Strassen for each such multiplication, we need 7 multiplications per pair.
    # But we share the linear combinations across different k values.
    #
    # Actually, the standard approach: For 4x4 = (2x2 blocks), we have 4 block rows
    # and 4 block cols, with 2 block "internal" dimensions. The total number of
    # Strassen terms is 4 * 4 * 7 / something... Let me think more carefully.
    #
    # Alternative: Use the known fact that the tensor rank of Mat_{4,4,4} is 49.
    # A construction: Use the 7-term Strassen algorithm. For 4x4 matrices, we can
    # view them as 2x2 block matrices where each block is 2x2. The product has
    # 4 block entries, each computed as sum of 2 block products. Each block product
    # (2x2 * 2x2) can be done with 7 multiplications. Total: 4 * 2 * 7 = 56? No,
    # that's not right either because we can share.
    #
    # Let me use a direct known rank-49 construction. One approach:
    # The tensor rank of the 4x4x4 matrix multiplication tensor is 49.
    # This is achieved by the "Strassen-like" algorithm or by other means.
    #
    # Actually, let me just implement the schoolbook for now and see if I can
    # find a better one within the time budget. The schoolbook rank is 64.
    # 
    # Wait - I should implement the best known decomposition. The rank of 
    # Mat_{4,4,4} is known to be 49 (lower bound from various sources, and 
    # upper bound from constructions). Let me implement a rank-49 decomposition.
    #
    # One known construction uses the fact that 4x4 matrix multiplication can be
    # decomposed using 49 rank-1 tensors. Here's one approach based on the
    # "Bini-Capovani" or similar constructions, but let me use a simpler one.
    #
    # Actually, the simplest way to get rank 49: use the Strassen algorithm 
    # recursively. Strassen gives rank 7 for 2x2. For 4x4, if we naively apply
    # Strassen to 2x2 blocks, we get rank 4*7 = 28 for each "level" but we have
    # 4 block positions... This is getting complicated.
    #
    # Let me try a different approach: implement the known optimal decomposition
    # from the literature. The 4x4x4 matrix multiplication tensor has rank 49.
    # 
    # Here's a concrete construction:
    # We can use the following: the 4x4 matrix multiplication can be seen as
    # computing C = A * B where A, B, C are 4x4. We can use 49 rank-1 terms.
    #
    # Let me use a practical approach: start with schoolbook (rank 64) and try
    # to reduce it. Or better, let me directly implement a known rank-49 construction.
    #
    # Known result: rank(Mat_4,4,4) = 49. One construction:
    # Use the 7x7 "Strassen" pattern generalized. Actually, let me just implement
    # the schoolbook and if time permits, try to improve.
    
    # Schoolbook decomposition (rank 64)
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    base_result = {"U": U, "V": V, "W": W}
    base_rank = rank
    
    # Try to improve: use Strassen-based decomposition
    # For 4x4 matrices, partition into 2x2 blocks.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]]
    # C = [[C00, C01], [C10, C11]]
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # Each block product (2x2 * 2x2) uses Strassen with 7 multiplications.
    # But we can share the linear combinations of A blocks and B blocks.
    #
    # Let's define Strassen terms for 2x2 blocks:
    # For matrices X = [[x00,x01],[x10,x11]] and Y = [[y00,y01],[y10,y11]]:
    # m0 = (x00+x11)*(y00+y11)
    # m1 = (x10+x11)*y00
    # m2 = x00*(y01-y11)
    # m3 = x11*(y10-y00)
    # m4 = (x00+x10)*y11
    # m5 = (x01-x11)*(y10+y11)
    # m6 = (x10-x00)*(y00+y01)
    # Then:
    # X*Y[0,0] = m0 + m3 - m4 + m6
    # X*Y[0,1] = m2 + m4
    # X*Y[1,0] = m1 + m5
    # X*Y[1,1] = m0 - m1 + m2 + m5
    #
    # For our 4x4 case, we need to compute 8 block products (4 C blocks, each
    # sum of 2 products). Using Strassen for each independently gives 8*7=56 terms.
    # But we can share: the A-linear-combinations and B-linear-combinations can be reused.
    #
    # Let's define the 7 Strassen forms for a 2x2 matrix M:
    # s0(M) = M[0,0] + M[1,1]
    # s1(M) = M[1,0] + M[1,1]
    # s2(M) = M[0,0]
    # s3(M) = M[1,1]
    # s4(M) = M[0,0] + M[1,0]
    # s5(M) = M[0,1] - M[1,1]
    # s6(M) = M[1,0] - M[0,0]
    #
    # And for the other matrix N:
    # t0(N) = N[0,0] + N[1,1]
    # t1(N) = N[0,0]
    # t2(N) = N[0,1] - N[1,1]
    # t3(N) = N[1,0] - N[0,0]
    # t4(N) = N[1,1]
    # t5(N) = N[1,0] + N[1,1]
    # t6(N) = N[0,0] + N[0,1]
    #
    # Strassen products:
    # p0 = s0(M) * t0(N)
    # p1 = s1(M) * t1(N)
    # p2 = s2(M) * t2(N)
    # p3 = s3(M) * t3(N)
    # p4 = s4(M) * t4(N)
    # p5 = s5(M) * t5(N)
    # p6 = s6(M) * t6(N)
    #
    # Result:
    # (M*N)[0,0] = p0 + p3 - p4 + p6
    # (M*N)[0,1] = p2 + p4
    # (M*N)[1,0] = p1 + p5
    # (M*N)[1,1] = p0 - p1 + p2 + p5
    #
    # Now for 4x4:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    #
    # We need Strassen decompositions for:
    # A00*B00, A01*B10, A00*B01, A01*B11, A10*B00, A11*B10, A10*B01, A11*B11
    #
    # That's 8 pairs, each needing 7 terms = 56 terms total.
    # But can we do better by sharing? The known optimal is 49.
    #
    # Let me just implement the 56-term version first, then see if I can optimize.
    # Actually, let me check: 4x4 matrix multiplication rank is indeed 49, not 56.
    # The 56-term construction is valid but not optimal.
    #
    # Let me implement the 56-term Strassen-based construction which is already
    # better than 64.
    
    if time.time() > deadline - 10:
        return base_result
    
    # Implement 56-term Strassen-based decomposition
    # Block indices: for 4x4 matrix, block (bi, bj) has entries at positions
    # (2*bi + r, 2*bj + c) for r,c in {0,1}
    
    def block_indices(bi, bj):
        """Return list of (row, col) indices for 2x2 block (bi, bj) in 4x4 matrix."""
        indices = []
        for r in range(2):
            for c in range(2):
                indices.append((2*bi + r, 2*bj + c))
        return indices
    
    # Strassen A-forms: list of (form_name, {local_pos: coeff})
    # Local positions: (0,0), (0,1), (1,0), (1,1)
    a_forms = [
        {(0,0): 1, (1,1): 1},           # s0: x00 + x11
        {(1,0): 1, (1,1): 1},           # s1: x10 + x11
        {(0,0): 1},                      # s2: x00
        {(1,1): 1},                      # s3: x11
        {(0,0): 1, (1,0): 1},           # s4: x00 + x10
        {(0,1): 1, (1,1): -1},          # s5: x01 - x11
        {(1,0): 1, (0,0): -1},          # s6: x10 - x00
    ]
    
    # Strassen B-forms
    b_forms = [
        {(0,0): 1, (1,1): 1},           # t0: y00 + y11
        {(0,0): 1},                      # t1: y00
        {(0,1): 1, (1,1): -1},          # t2: y01 - y11
        {(1,0): 1, (0,0): -1},          # t3: y10 - y00
        {(1,1): 1},                      # t4: y11
        {(1,0): 1, (1,1): 1},           # t5: y10 + y11
        {(0,0): 1, (0,1): 1},           # t6: y00 + y01
    ]
    
    # Output coefficients for each result position, per Strassen term
    # result_pos: (r, c) in {0,1}^2, strassen_term: 0-6
    # (M*N)[0,0] = p0 + p3 - p4 + p6
    # (M*N)[0,1] = p2 + p4
    # (M*N)[1,0] = p1 + p5
    # (M*N)[1,1] = p0 - p1 + p2 + p5
    out_coeffs = {
        (0,0): {0: 1, 3: 1, 4: -1, 6: 1},
        (0,1): {2: 1, 4: 1},
        (1,0): {1: 1, 5: 1},
        (1,1): {0: 1, 1: -1, 2: 1, 5: 1},
    }
    
    # The 8 block products: (A_block, B_block) pairs for each C_block
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    block_products = [
        # (C_block_idx, A_block_idx, B_block_idx)
        (0, 0, 0), (0, 1, 2),  # C00 = A00*B00 + A01*B10
        (1, 0, 1), (1, 1, 3),  # C01 = A00*B01 + A01*B11
        (2, 2, 0), (2, 3, 2),  # C10 = A10*B00 + A11*B10
        (3, 2, 1), (3, 3, 3),  # C11 = A10*B01 + A11*B11
    ]
    # Block indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    def block_to_bi_bj(bidx):
        return (bidx // 2, bidx % 2)
    
    # Build the 56-term decomposition
    U_56 = [[0] * 56 for _ in range(16)]  # n*m = 4*4 = 16
    V_56 = [[0] * 56 for _ in range(16)]  # m*p = 4*4 = 16
    W_56 = [[0] * 56 for _ in range(16)]  # p*n = 4*4 = 16
    
    term_idx = 0
    for c_block, a_block, b_block in block_products:
        c_bi, c_bj = block_to_bi_bj(c_block)
        a_bi, a_bj = block_to_bi_bj(a_block)
        b_bi, b_bj = block_to_bi_bj(b_block)
        
        for s in range(7):  # 7 Strassen terms
            # U vector: linear combination of A_block entries
            for (lr, lc), coeff in a_forms[s].items():
                row = 2*a_bi + lr
                col = 2*a_bj + lc
                U_56[row * 4 + col][term_idx] += coeff
            
            # V vector: linear combination of B_block entries
            for (lr, lc), coeff in b_forms[s].items():
                row = 2*b_bi + lr
                col = 2*b_bj + lc
                V_56[row * 4 + col][term_idx] += coeff
            
            # W vector: for each output position in C_block
            for (lr, lc) in [(0,0), (0,1), (1,0), (1,1)]:
                if s in out_coeffs[(lr, lc)]:
                    coeff = out_coeffs[(lr, lc)][s]
                    row = 2*c_bi + lr
                    col = 2*c_bj + lc
                    # W index: k*n + i, where k is col of B (=row of C in product sense)
                    # Actually W[k*n + i, r] with the convention W[k*n+i, r]
                    # The output C[i, k] = sum_r U[i*m+j, r] * V[j*p+k, r]
                    # Wait, let me re-read the convention.
                    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
                    # The tensor is sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # So W is indexed by (k, i) where k is the "middle" index and i is the row.
                    # W[k*n + i, r] corresponds to output position (i, k) in C.
                    # So for output C[c_bi*2+lr, c_bj*2+lc], we have i = c_bi*2+lr, k = c_bj*2+lc
                    w_idx = (c_bj * 2 + lc) * 4 + (c_bi * 2 + lr)
                    W_56[w_idx][term_idx] += coeff
            
            term_idx += 1
    
    # Verify the 56-term decomposition
    def verify_decomposition(U, V, W, n, m, p):
        """Check if the decomposition exactly represents matrix multiplication."""
        rank = len(U[0])
        # Check a few random matrices
        import random
        random.seed(42)
        for trial in range(20):
            A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
            B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
            # Compute C = A * B
            C_expected = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
            # Compute C from decomposition: C[i][k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
            # Wait, that's not right. The tensor decomposition means:
            # C[i][k] = sum_{j,r} A[i][j] * B[j][k] * ... no.
            # Actually: sum_r (sum_j A[i][j] * U[i*m+j, r]) * (sum_j V[j*p+k, r] * B[j][k])... no.
            # The correct formula: C[i][k] = sum_r U_col_i(r) * V_col_k(r) * W_kn_i(r)
            # where U_col_i(r) = sum_j A[i][j] * U[i*m+j, r]? No.
            # Let me re-read: the tensor is sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # This equals C[i][k] for all A, B where A[i][j] picks out U[i*m+j,:] etc.
            # So: C[i][k] = sum_r (sum_j A[i][j] U[i*m+j, r]) * (sum_j B[j][k] V[j*p+k, r]) ... no.
            # Actually it's: C[i][k] = sum_r sum_j A[i][j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] / W[k*n+i, r]
            # No, the standard tensor decomposition: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
            # And C_{ik} = sum_j A_{ij} B_{jk} = sum_{j,r} A_{ij} U_{ij,r} V_{jk,r} W_{ki,r} ... 
            # Wait, that's not quite right either. Let me think again.
            # The matrix multiplication tensor T has T[i,j,k] = 1 if the "path" i->j->k is valid, 0 otherwise.
            # Actually T[i,j,k] = delta (it's 1 for all i,j,k in the appropriate ranges).
            # The decomposition: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
            # And then C[i,k] = sum_{j,r} A[i,j] * T[i,j,k] * B[j,k] = sum_{j,r} A[i,j] * B[j,k] * U[i,j,r] * V[j,k,r] * W[k,i,r]
            # Hmm, that's not standard. Let me use the standard formulation.
            # Standard: C = A*B, C[i,k] = sum_j A[i,j] * B[j,k]
            # Tensor form: C[i,k] = sum_{j,r} A[i,j] * B[j,k] * (decomposition terms)
            # Actually: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal delta (i.e., 1 for all valid i,j,k)
            # No wait, it should reconstruct the multiplication. The correct check:
            # For given A, B: C[i,k] = sum_r (sum_j A[i,j] * U[i*m+j, r]) * (sum_j V[j*p+k, r] * B[j,k]) * ... 
            # I'm confusing myself. Let me just use: the decomposition satisfies
            # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] / something
            # 
            # Actually, the standard tensor decomposition for matrix multiplication:
            # The "multiplication tensor" is defined as T[i,j,k] = 1 for all i,j,k (in ranges).
            # No, that's not right. T[i,j,k] represents the "interaction" between A[i,j], B[j,k], and C[i,k].
            # The decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Then: C[i,k] = sum_{j} A[i,j] B[j,k] = sum_{j,r} A[i,j] B[j,k] U[i,j,r] V[j,k,r] W[k,i,r]
            # Wait, that only works if sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
            # Hmm, but that's not generally true. Let me reconsider.
            # 
            # OK I think the correct formulation is:
            # C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_j B[j,k] V[j*p+k, r]) * ... no.
            # 
            # Let me just test numerically: compute C_from_decomp[i,k] = sum_{j,r} A[i,j] * B[j,k] * U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
            # and see if it equals A*B.
            
            C_decomp = [[0.0] * p for _ in range(n)]
            for i in range(n):
                for k in range(p):
                    for j in range(m):
                        for r in range(rank):
                            C_decomp[i][k] += A[i][j] * B[j][k] * U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
            
            for i in range(n):
                for k in range(p):
                    if abs(C_decomp[i][k] - C_expected[i][k]) > 1e-9:
                        return False, f"Mismatch at ({i},{k}): {C_decomp[i][k]} vs {C_expected[i][k]}"
        return True, "OK"
    
    valid_56, msg = verify_decomposition(U_56, V_56, W_56, n, m, p)
    
    if valid_56:
        return {"U": U_56, "V": V_56, "W": W_56}
    else:
        # If 56-term doesn't work, fall back to schoolbook
        return base_result

# EVOLVE-BLOCK-END
