"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-49 Strassen decomposition for 4x4 matrix multiplication.
    # This is the known optimal rank for 4x4 matrix multiplication over Q/C.
    # The decomposition uses a 2x2 Strassen block structure applied at multiple levels.
    n, m, p = DIMENSIONS  # 4, 4, 4
    rank = 49
    U = [[0] * rank for _ in range(n * m)]  # 16 x 49
    V = [[0] * rank for _ in range(m * p)]  # 16 x 49
    W = [[0] * rank for _ in range(p * n)]  # 16 x 49

    # Strassen's algorithm for 2x2 matrix multiplication:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B21 - B22)
    # M4 = A22(B12 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4, we partition into 2x2 blocks of size 2x2:
    # A = [[A00, A01], [A10, A11]] where each Ai is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bb is 2x2
    # C = A*B = [[C00, C01], [C10, C11]] where each Cc is 2x2

    # We need to decompose the product of 2x2 blocks using Strassen.
    # Each 2x2 block product can be done with 7 multiplications (Strassen),
    # but we need to be careful about how the blocks interact.

    # Actually, for 4x4 using Strassen, we partition into 2x2 blocks and apply
    # Strassen at the block level. This gives us 7 "big" multiplications, each
    # of which is a product of 2x2 matrices. But each 2x2 product can itself be
    # done with 7 multiplications, giving 7*7 = 49 total multiplications.

    # Let me define the Strassen factors for 2x2 blocks.
    # For a 2x2 block product P = X*Y where X, Y are 2x2:
    # The Strassen decomposition uses 7 rank-1 terms.

    # Let me work out the explicit U, V, W matrices for rank 49.

    # The 4x4 matrix A has entries A[i,j] for i,j in {0,1,2,3}.
    # Block partition: A00 = A[0:2, 0:2], A01 = A[0:2, 2:4], A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
    # Similarly for B.
    # C[i,k] for i,k in {0,1,2,3}.
    # C00 = C[0:2, 0:2], C01 = C[0:2, 2:4], C10 = C[2:4, 0:2], C11 = C[2:4, 2:4]

    # Using Strassen at the block level:
    # S1 = (A00 + A11)(B00 + B11)
    # S2 = (A10 + A11)B00
    # S3 = A00(B10 - B11)
    # S4 = A11(B01 - B00)
    # S5 = (A00 + A01)B11
    # S6 = (A10 - A00)(B00 + B01)
    # S7 = (A01 - A11)(B10 + B11)
    # C00 = S1 + S4 - S5 + S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S2 + S3 + S6

    # Now each Sj is a product of two 2x2 matrices. For example, S1 = (A00+A11)(B00+B11).
    # Let X = A00 + A11 (2x2) and Y = B00 + B11 (2x2). Then S1 = X*Y.
    # The product X*Y of two 2x2 matrices can be decomposed into 7 rank-1 terms using Strassen.

    # For a general 2x2 matrix product P = X*Y where X = [[x00,x01],[x10,x11]] and Y = [[y00,y01],[y10,y11]]:
    # P00 = x00*y00 + x01*y10
    # P01 = x00*y01 + x01*y11
    # P10 = x10*y00 + x11*y10
    # P11 = x10*y01 + x11*y11
    # Using Strassen:
    # T1 = (x00+x11)(y00+y11)
    # T2 = (x10+x11)y00
    # T3 = x00(y10-y11)
    # T4 = x11(y01-y00)
    # T5 = (x00+x01)y11
    # T6 = (x10-x00)(y00+y01)
    # T7 = (x01-x11)(y10+y11)
    # P00 = T1 + T4 - T5 + T7
    # P01 = T3 + T5
    # P10 = T2 + T4
    # P11 = T1 - T2 + T3 + T6

    # Now I need to build U, V, W for the overall rank-49 decomposition.
    # The 49 rank-1 terms correspond to (s, t) where s in {0..6} (which Strassen block)
    # and t in {0..6} (which Strassen term within that block).

    # Let me define the contributions.
    # For each (s, t), we have a rank-1 term contributing to C.
    # The contribution to C[i,k] is U[i*4+j, s*7+t] * V[j*4+k, s*7+t] * W[k*4+i, s*7+t]
    # summed over j.

    # Actually, the tensor decomposition for matrix multiplication is:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # In the rank-r decomposition:
    # C[i,k] = sum_{c=0}^{r-1} U[i*4+j_c, c] * V[j_c*4+k, c] * W[k*4+i, c]
    # Wait, that's not quite right. The standard form is:
    # C[i,k] = sum_{c=0}^{r-1} sum_j U[i,j,c] * V[j,k,c] * W[k,i,c]
    # But the interface uses flattened indices.

    # Let me reconsider. The interface says:
    # U has shape (n*m, rank) = (16, 49)
    # V has shape (m*p, rank) = (16, 49)
    # W has shape (p*n, rank) = (16, 49)
    # The decomposition is:
    # C[i,k] = sum_{j=0}^{3} sum_{c=0}^{48} U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]

    # So for each rank-1 term c, we have:
    # U[:, c] is a vector of length 16 (indexed by i*4+j)
    # V[:, c] is a vector of length 16 (indexed by j*4+k)
    # W[:, c] is a vector of length 16 (indexed by k*4+i)

    # And the contribution to C[i,k] from term c is:
    # sum_j U[i*4+j, c] * V[j*4+k, c] * W[k*4+i, c]

    # Let me build this carefully.

    # I'll define helper functions to get/set entries.
    def u_idx(i, j): return i * 4 + j
    def v_idx(j, k): return j * 4 + k
    def w_idx(k, i): return k * 4 + i

    # Now let me enumerate the 49 terms.
    # Term (s, t) where s in 0..6, t in 0..6, col = s*7 + t.

    # For s=0: S1 = (A00+A11)(B00+B11)
    #   X = A00 + A11, Y = B00 + B11
    #   X[i,j] for i,j in {0,1}:
    #     X[0,0] = A[0,0] + A[2,2]
    #     X[0,1] = A[0,1] + A[2,3]
    #     X[1,0] = A[1,0] + A[3,2]
    #     X[1,1] = A[1,1] + A[3,3]
    #   Y[i,j] for i,j in {0,1}:
    #     Y[0,0] = B[0,0] + B[2,2]
    #     Y[0,1] = B[0,1] + B[2,3]
    #     Y[1,0] = B[1,0] + B[3,2]
    #     Y[1,1] = B[1,1] + B[3,3]

    # The Strassen terms for S1 (with X, Y as above):
    # t=0: T1 = (x00+x11)(y00+y11)
    # t=1: T2 = (x10+x11)y00
    # t=2: T3 = x00(y10-y11)
    # t=3: T4 = x11(y01-y00)
    # t=4: T5 = (x00+x01)y11
    # t=5: T6 = (x10-x00)(y00+y01)
    # t=6: T7 = (x01-x11)(y10+y11)

    # For each t, the contribution to the 2x2 block product P = X*Y is:
    # P[0,0] gets coefficient from T_t
    # P[0,1] gets coefficient from T_t
    # P[1,0] gets coefficient from T_t
    # P[1,1] gets coefficient from T_t

    # The coefficients for P[i,j] from T_t are:
    # t=0: P[0,0]+=1, P[1,1]+=1
    # t=1: P[1,0]+=1
    # t=2: P[0,1]+=1
    # t=3: P[0,1]+=1, P[0,0]+=1
    # t=4: P[0,1]+=1, P[0,0]-=1
    # t=5: P[1,1]+=1
    # t=6: P[0,0]+=1, P[1,1]+=1
    # Wait, let me recompute.
    # P00 = T1 + T4 - T5 + T7
    # P01 = T3 + T5
    # P10 = T2 + T4
    # P11 = T1 - T2 + T3 + T6

    # So the coefficients are:
    # T0 (t=0): P00+=1, P11+=1
    # T1 (t=1): P10+=1
    # T2 (t=2): P01+=1
    # T3 (t=3): P00+=1, P10+=1
    # T4 (t=4): P00-=1, P01+=1
    # T5 (t=5): P11+=1
    # T6 (t=6): P00+=1, P11+=1

    # Now for the overall C, S1 contributes to C00 and C11:
    # C00 = S1 + S4 - S5 + S7 => S1 contributes +1 to C00
    # C11 = S1 - S2 + S3 + S6 => S1 contributes +1 to C11
    # So S1's P[i,j] (which is a 2x2 block) contributes to C[i*2:i*2+2, j*2:j*2+2] with coefficient +1
    # for (i,j) in {(0,0), (1,1)}.

    # This is getting complex. Let me just directly construct U, V, W by computing
    # the coefficients for each of the 49 rank-1 terms.

    # I'll use a more systematic approach. Let me define the left and right factors
    # for each of the 7 Strassen blocks, then for each block, define the 7 inner
    # Strassen terms.

    # Actually, let me just hardcode the U, V, W matrices. This is a known result.
    # The rank-49 decomposition of 4x4 matrix multiplication is well-known.

    # Let me construct it step by step.
    # I'll define for each column c (0 to 48), the vectors U[:,c], V[:,c], W[:,c].

    # col = s*7 + t where s in 0..6 (outer Strassen), t in 0..6 (inner Strassen)

    # For each (s, t), I need to determine:
    # - The coefficients of A[i,j] in the left factor (for U)
    # - The coefficients of B[j,k] in the right factor (for V)
    # - The coefficients in W (for the output block)

    # Let me define the outer Strassen left factors (as 2x2 block matrices):
    # L0 = A00 + A11
    # L1 = A10 + A11
    # L2 = A00
    # L3 = A11
    # L4 = A00 + A01
    # L5 = A10 - A00
    # L6 = A01 - A11

    # And right factors:
    # R0 = B00 + B11
    # R1 = B00
    # R2 = B10 - B11
    # R3 = B01 - B00
    # R4 = B11
    # R5 = B00 + B01
    # R6 = B10 + B11

    # And output coefficients (which blocks of C each S contributes to):
    # S0: C00 += 1, C11 += 1
    # S1: C10 += 1
    # S2: C01 += 1
    # S3: C00 += 1, C10 += 1
    # S4: C00 -= 1, C01 += 1
    # S5: C11 += 1
    # S6: C00 += 1, C11 += 1

    # Now for each outer factor (Ls, Rs), I apply inner Strassen to get 7 rank-1 terms.
    # For the inner Strassen on Ls * Rs (both 2x2 block matrices):
    # The 7 inner terms are:
    # t=0: (Ls[0,0]+Ls[1,1])(Rs[0,0]+Rs[1,1])
    # t=1: (Ls[1,0]+Ls[1,1])Rs[0,0]
    # t=2: Ls[0,0](Rs[1,0]-Rs[1,1])
    # t=3: Ls[1,1](Rs[0,1]-Rs[0,0])
    # t=4: (Ls[0,0]+Ls[0,1])Rs[1,1]
    # t=5: (Ls[1,0]-Ls[0,0])(Rs[0,0]+Rs[0,1])
    # t=6: (Ls[0,1]-Ls[1,1])(Rs[1,0]+Rs[1,1])

    # And the inner Strassen output coefficients:
    # t=0: P00+=1, P11+=1
    # t=1: P10+=1
    # t=2: P01+=1
    # t=3: P00+=1, P10+=1
    # t=4: P00-=1, P01+=1
    # t=5: P11+=1
    # t=6: P00+=1, P11+=1

    # Now I need to express Ls[i,j] and Rs[i,j] in terms of A and B entries.
    # Ls is a 2x2 block matrix, so Ls[i,j] is itself a 2x2 matrix.
    # Similarly for Rs.

    # Let me define the 2x2 block entries.
    # For a 4x4 matrix M, M00 = M[0:2,0:2], M01 = M[0:2,2:4], M10 = M[2:4,0:2], M11 = M[2:4,2:4]
    # M00[a,b] = M[a*4+b] for a,b in {0,1}
    # M01[a,b] = M[a*4+b+2] for a,b in {0,1}
    # M10[a,b] = M[(a+2)*4+b] for a,b in {0,1}
    # M11[a,b] = M[(a+2)*4+b+2] for a,b in {0,1}

    # This is getting very detailed. Let me just write out the full U, V, W matrices
    # by computing each entry.

    # I'll use a helper to build the matrices.
    import time
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit

    # Define the 7 outer Strassen left factors as 4x4 matrices (sparse)
    # Each is represented as a dict mapping (i,j) -> coefficient
    # Actually, let me represent them as 4x4 matrices with 0/1/-1 entries.

    def block_sum(b1, b2):
        """Sum two 2x2 block matrices, returning a 4x4 matrix."""
        result = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                result[i][j] = b1[i][j] + b2[i][j]
        return result

    def block_diff(b1, b2):
        """Difference of two 2x2 block matrices."""
        result = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                result[i][j] = b1[i][j] - b2[i][j]
        return result

    # Define A block matrices (as 4x4 with 1s in the appropriate blocks)
    A00 = [[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]
    A01 = [[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]]
    A10 = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,0]]
    A11 = [[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]]

    B00 = [[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]
    B01 = [[0,0,1,1],[0,0,1,1],[0,0,0,0],[0,0,0,0]]
    B10 = [[0,0,0,0],[0,0,0,0],[1,1,0,0],[1,1,0,0]]
    B11 = [[0,0,0,0],[0,0,0,0],[0,0,1,1],[0,0,1,1]]

    # Outer Strassen left factors
    L0 = block_sum(A00, A11)
    L1 = block_sum(A10, A11)
    L2 = A00
    L3 = A11
    L4 = block_sum(A00, A01)
    L5 = block_diff(A10, A00)
    L6 = block_diff(A01, A11)

    # Outer Strassen right factors
    R0 = block_sum(B00, B11)
    R1 = B00
    R2 = block_diff(B10, B11)
    R3 = block_diff(B01, B00)
    R4 = B11
    R5 = block_sum(B00, B01)
    R6 = block_sum(B10, B11)

    L_factors = [L0, L1, L2, L3, L4, L5, L6]
    R_factors = [R0, R1, R2, R3, R4, R5, R6]

    # Output coefficients: for each outer term s, which C blocks does it contribute to, and with what coefficient?
    # C00, C01, C10, C11 are 2x2 blocks.
    # S0 (s=0): C00 += 1, C11 += 1
    # S1 (s=1): C10 += 1
    # S2 (s=2): C01 += 1
    # S3 (s=3): C00 += 1, C10 += 1
    # S4 (s=4): C00 -= 1, C01 += 1
    # S5 (s=5): C11 += 1
    # S6 (s=6): C00 += 1, C11 += 1

    # Represent as dict: s -> {(block_row, block_col): coeff}
    # block_row in {0,1}, block_col in {0,1}
    output_coeffs = [
        {(0,0): 1, (1,1): 1},  # s=0
        {(1,0): 1},             # s=1
        {(0,1): 1},             # s=2
        {(0,0): 1, (1,0): 1},  # s=3
        {(0,0): -1, (0,1): 1}, # s=4
        {(1,1): 1},             # s=5
        {(0,0): 1, (1,1): 1},  # s=6
    ]

    # Inner Strassen: for a 2x2 block product X*Y (where X, Y are 2x2 block matrices,
    # i.e., 4x4 matrices with block structure), the 7 inner terms produce a 2x2 block result.
    # The inner Strassen terms t=0..6 have output coefficients:
    inner_output_coeffs = [
        {(0,0): 1, (1,1): 1},  # t=0
        {(1,0): 1},             # t=1
        {(0,1): 1},             # t=2
        {(0,0): 1, (1,0): 1},  # t=3
        {(0,0): -1, (0,1): 1}, # t=4
        {(1,1): 1},             # t=5
        {(0,0): 1, (1,1): 1},  # t=6
    ]

    # Now for each (s, t), I need to compute:
    # - The left factor for the t-th inner Strassen term applied to Ls
    # - The right factor for the t-th inner Strassen term applied to Rs
    # - The combined output coefficient

    # The inner Strassen left factors for a 2x2 block matrix X (4x4 with block structure):
    # t=0: X00 + X11
    # t=1: X10 + X11
    # t=2: X00
    # t=3: X11
    # t=4: X00 + X01
    # t=5: X10 - X00
    # t=6: X01 - X11
    # where X00, X01, X10, X11 are the 2x2 blocks of X.

    # Similarly for the right factors.

    # Let me compute the inner left and right factors for each (s, t).
    def inner_left_factor(X, t):
        """Compute the left factor for the t-th inner Strassen term."""
        X00 = [[X[i][j] for j in range(2)] for i in range(2)]
        X01 = [[X[i][j+2] for j in range(2)] for i in range(2)]
        X10 = [[X[i+2][j] for j in range(2)] for i in range(2)]
        X11 = [[X[i+2][j+2] for j in range(2)] for i in range(2)]

        def sum2(M1, M2):
            return [[M1[i][j] + M2[i][j] for j in range(2)] for i in range(2)]
        def diff2(M1, M2):
            return [[M1[i][j] - M2[i][j] for j in range(2)] for i in range(2)]

        if t == 0: return sum2(X00, X11)
        elif t == 1: return sum2(X10, X11)
        elif t == 2: return X00
        elif t == 3: return X11
        elif t == 4: return sum2(X00, X01)
        elif t == 5: return diff2(X10, X00)
        elif t == 6: return diff2(X01, X11)

    def inner_right_factor(Y, t):
        """Compute the right factor for the t-th inner Strassen term."""
        Y00 = [[Y[i][j] for j in range(2)] for i in range(2)]
        Y01 = [[Y[i][j+2] for j in range(2)] for i in range(2)]
        Y10 = [[Y[i+2][j] for j in range(2)] for i in range(2)]
        Y11 = [[Y[i+2][j+2] for j in range(2)] for i in range(2)]

        def sum2(M1, M2):
            return [[M1[i][j] + M2[i][j] for j in range(2)] for i in range(2)]
        def diff2(M1, M2):
            return [[M1[i][j] - M2[i][j] for j in range(2)] for i in range(2)]

        if t == 0: return sum2(Y00, Y11)
        elif t == 1: return Y00
        elif t == 2: return diff2(Y10, Y11)
        elif t == 3: return diff2(Y01, Y00)
        elif t == 4: return Y11
        elif t == 5: return sum2(Y00, Y01)
        elif t == 6: return sum2(Y10, Y11)

    # Now build U, V, W.
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            if time.time() > deadline:
                break

            # Left factor for inner Strassen
            LF = inner_left_factor(L_factors[s], t)  # 2x2
            # Right factor for inner Strassen
            RF = inner_right_factor(R_factors[s], t)  # 2x2
            # Inner output coefficients
            inner_coeffs = inner_output_coeffs[t]  # {(br,bc): coeff}
            # Outer output coefficients
            outer_coeffs = output_coeffs[s]  # {(br,bc): coeff}

            # The left factor LF is a 2x2 matrix representing a 4x4 matrix with block structure.
            # LF[0,0] corresponds to A entries in block (0,0), etc.
            # Specifically, LF[a,b] is the coefficient for the (a,b)-th entry of the 2x2 block.
            # The 4x4 left factor has entries:
            # L4x4[i,j] where i = br*2 + a, j = bc*2 + b
            # and L4x4[i,j] = LF[a,b] if the block (br,bc) is selected.

            # Actually, LF is already a 2x2 matrix, and it represents the coefficients
            # for the 2x2 block. The full 4x4 left factor is:
            # For block (br, bc) with coefficient from outer factor L_factors[s][br][bc]:
            # The left factor entry at (br*2+a, bc*2+b) = L_factors[s][br][bc] * LF[a][b]

            # Similarly for the right factor.

            # Let me compute U[:, col]:
            # U[i*4+j, col] = sum over blocks of L_factors[s][block] * LF[a][b]
            # where i = br*2+a, j = bc*2+b
            for br in range(2):
                for bc in range(2):
                    l_block_coeff = L_factors[s][br][bc]
                    if l_block_coeff == 0:
                        continue
                    for a in range(2):
                        for b in range(2):
                            lf_coeff = LF[a][b]
                            if lf_coeff == 0:
                                continue
                            i = br * 2 + a
                            j = bc * 2 + b
                            U[i * 4 + j][col] += l_block_coeff * lf_coeff

            # V[:, col]:
            # V[j*4+k, col] = sum over blocks of R_factors[s][block] * RF[a][b]
            # where j = br*2+a, k = bc*2+b
            for br in range(2):
                for bc in range(2):
                    r_block_coeff = R_factors[s][br][bc]
                    if r_block_coeff == 0:
                        continue
                    for a in range(2):
                        for b in range(2):
                            rf_coeff = RF[a][b]
                            if rf_coeff == 0:
                                continue
                            j = br * 2 + a
                            k = bc * 2 + b
                            V[j * 4 + k][col] += r_block_coeff * rf_coeff

            # W[:, col]:
            # W[k*4+i, col] = coefficient for C[i,k]
            # The output block (br, bc) from inner Strassen contributes to C block (br, bc)
            # with coefficient inner_coeffs[(br,bc)]
            # And the outer term s contributes to C block (br, bc) with coefficient outer_coeffs[(br,bc)]
            # So the combined coefficient for C block (br, bc) is outer_coeffs.get((br,bc), 0) * inner_coeffs.get((br,bc), 0)
            # And within C block (br, bc), the entry C[br*2+a, bc*2+b] gets this coefficient.
            for br in range(2):
                for bc in range(2):
                    outer_c = outer_coeffs.get((br, bc), 0)
                    inner_c = inner_coeffs.get((br, bc), 0)
                    if outer_c == 0 or inner_c == 0:
                        continue
                    coeff = outer_c * inner_c
                    for a in range(2):
                        for b in range(2):
                            i = br * 2 + a
                            k = bc * 2 + b
                            W[k * 4 + i][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
