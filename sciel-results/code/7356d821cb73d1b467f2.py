import time
import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s
    n, m, p = DIMENSIONS  # 4, 4, 9

    # ---------- Schoolbook fallback (rank 144), guaranteed correct ----------
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---------- Strassen 4x4x4 rank-49 decomposition generator ----------
    # We represent the 4x4 A and 4x4 B as 2x2 blocks of 2x2 matrices.
    # A-block (r,c) = A[2r:2r+2, 2c:2c+2], B-block (r,c) = B[2r:2r+2, 2c:2c+2].
    # C-block (r,c) = C[2r:2r+2, 2c:2c+2].
    #
    # Each rank-1 term is (u, v, w) where:
    #   u: length-16 vector (flattened 4x4 A) with integer entries
    #   v: length-16 vector (flattened 4x4 B) with integer entries
    #   w: length-16 vector (flattened 4x4 C) with integer entries
    # such that u . A_flat * v . B_flat ... actually the tensor term is u ⊗ v ⊗ w.
    #
    # Strassen 4x4: 7 products S1..S7, each = X * Y where X,Y are 2x2 matrices
    # of 2x2 blocks (linear combos of A-blocks / B-blocks).
    # Each X*Y is expanded via Strassen 2x2 into 7 rank-1 terms M1..M7.
    #
    # Strassen 4x4 combinations (X, Y) as 2x2 matrices of block-coefficient
    # matrices. Each block-coefficient matrix is 2x2 with entries in {0,1,-1}
    # indicating which 2x2 sub-block of A (or B) to add.
    #
    # We define X and Y as 2x2 arrays, each element a 2x2 numpy array of
    # coefficients for the 4 scalar entries of the corresponding 2x2 block.
    #
    # Let's define block access: for A, block (br,bc) occupies rows 2br:2br+2,
    # cols 2bc:2bc+2. Within a block, entry (a,b) is A[2br+a, 2bc+b].

    # Strassen 4x4: S1..S7. For each, X and Y are 2x2 block-matrices.
    # X is built from A-blocks, Y from B-blocks.
    # Block-matrix (r,c) holds a 2x2 coefficient matrix for the 2x2 block.
    #
    # S1 = (A00+A11)(B00+B11): X = [[A00,0],[0,A11]] as block-matrix? No.
    # Actually X is a 2x2 matrix whose (0,0) entry = A00, (1,1) entry = A11,
    # off-diagonals = 0. So X = [[A00, 0],[0, A11]] where each entry is a 2x2 block.
    # Similarly Y = [[B00, 0],[0, B11]].
    #
    # S2 = (A10+A11)B00: X = [[0,0],[A10, A11]], Y = [[B00,0],[0,0]].
    # S3 = A00(B01-B11): X = [[A00,0],[0,0]], Y = [[0,B01],[0,-B11]].
    # S4 = A11(B10-B00): X = [[0,0],[0,A11]], Y = [[0,0],[B10,-B00]].
    # S5 = (A00+A01)B11: X = [[A00,A01],[0,0]], Y = [[0,0],[0,B11]].
    # S6 = (A10-A00)(B00+B01): X = [[-A00,0],[A10,0]], Y = [[B00,B01],[0,0]].
    # S7 = (A01-A11)(B10+B11): X = [[0,A01],[0,-A11]], Y = [[0,0],[B10,B11]].
    #
    # C reconstruction:
    # C00 = S1 - S4 + S6 - S7
    # C01 = S3 + S5
    # C10 = S2 + S4
    # C11 = S1 - S3 + S2 - S5
    #
    # For each Si = X*Y (2x2 block-matrix product), we expand via Strassen 2x2.
    # Strassen 2x2 on 2x2 matrices X (with entries being 2x2 blocks) and Y:
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)
    # P00 = M1 - M4 + M6 - M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M3 + M2 - M5
    #
    # Each Mj = Xj * Yj where Xj, Yj are 2x2 matrices (each entry a 2x2 block
    # combo). We expand Xj*Yj as a single rank-1 term in the 4x4x4 tensor:
    # Xj is a 2x2 matrix of 2x2 blocks (linear combos of A entries).
    # Yj is a 2x2 matrix of 2x2 blocks (linear combos of B entries).
    # The product Xj*Yj gives a 2x2 matrix of 2x2 blocks = a 4x4 matrix.
    # As a rank-1 tensor term: u ⊗ v ⊗ w where
    #   u[i,j] = coefficient of A[i,j] in Xj (flattened 4x4)
    #   v[i,j] = coefficient of B[i,j] in Yj (flattened 4x4)
    #   w[i,j] = coefficient of C[i,j] in (Xj*Yj) (flattened 4x4)
    #
    # Wait: the tensor term for "Xj * Yj" where Xj = sum u_a A_a, Yj = sum v_b B_b
    # is NOT a single rank-1 term in general. Xj*Yj = (sum u_a A_a)(sum v_b B_b)
    # = sum_{a,b} u_a v_b A_a B_b. That's a product of two linear combos, which
    # in the tensor decomposition corresponds to the OUTER product of the
    # coefficient vectors: the rank-1 term has u (coeffs on A), v (coeffs on B),
    # and w = the resulting C pattern = Xj*Yj evaluated symbolically.
    #
    # Actually for the tensor M (matrix mult), a rank-1 term is u⊗v⊗w where
    # M[i,j,k] = sum_t u[i,j]_t v[j,k]_t w[k,i]_t. Here for a 4x4x4 block,
    # a rank-1 term corresponds to: u = coeffs on the 16 A entries, v = coeffs
    # on the 16 B entries, w = coeffs on the 16 C entries, such that
    # C = (A*u_pattern)(B*v_pattern) = w_pattern, i.e., the 4x4 matrix
    # (sum_ij u_ij A_ij) (sum_jk v_jk B_jk) = sum_ki w_ki C_ki.
    #
    # For Xj*Yj: Xj is a 4x4 matrix = sum_{i,j} u_ij A[i,j] where u_ij is the
    # coefficient. Yj is a 4x4 matrix = sum_{j,k} v_jk B[j,k]. Then
    # Xj*Yj = (4x4 product) = sum_{i,k} (Xj*Yj)_{i,k} C[i,k]. So w_{i,k} =
    # (Xj*Yj)_{i,k} as a number. And u, v are the coefficient matrices.
    #
    # So for each of the 49 terms, we have:
    #   u: 4x4 matrix of coefficients (integers)
    #   v: 4x4 matrix of coefficients (integers)
    #   w: 4x4 matrix = u_mat @ v_mat (the symbolic product, integers)
    # and the term is u⊗v⊗w.
    #
    # The 49 terms come from 7 outer S's, each expanded to 7 inner M's.
    # For outer S, X and Y are 2x2 block-matrices. For inner Mj, Xj and Yj
    # are 2x2 matrices of 2x2 blocks. Xj is a 4x4 matrix (concatenating the
    # 2x2 blocks), Yj is a 4x4 matrix. u = Xj (as 4x4 coeff matrix), v = Yj,
    # w = Xj @ Yj.

    def make_block(r, c, sign=1):
        """Return a 2x2 coefficient matrix for A-block (r,c) with given sign."""
        B = np.zeros((2, 2), dtype=int)
        B[0, 0] = sign
        B[0, 1] = 0
        B[1, 0] = 0
        B[1, 1] = 0
        # Actually the block (r,c) is a 2x2 matrix; the coefficient matrix
        # should mark which scalar entries. For a full block with uniform sign,
        # the 2x2 coeff matrix is sign * I? No.
        # A-block (r,c) = A[2r:2r+2, 2c:2c+2] is a 2x2 matrix. When we write
        # X as a 4x4 coefficient matrix, the block (r,c) occupies rows 2r:2r+2,
        # cols 2c:2c+2, and the entries are 1 (or sign) for all 4 positions.
        B[:] = sign
        return B

    def blockmat(b00, b01, b10, b11):
        """Assemble a 4x4 coefficient matrix from four 2x2 blocks."""
        M = np.zeros((4, 4), dtype=int)
        M[0:2, 0:2] = b00
        M[0:2, 2:4] = b01
        M[2:4, 0:2] = b10
        M[2:4, 2:4] = b11
        return M

    def zero2():
        return np.zeros((2, 2), dtype=int)

    def blk(r, c, sign=1):
        """2x2 block coefficient matrix for block (r,c) with uniform sign."""
        return np.full((2, 2), sign, dtype=int)

    # Define the 7 outer Strassen-4x4 combinations.
    # Each is (X_blockmat, Y_blockmat) where each is a 2x2 array of 2x2 blocks.
    # We'll represent X as a 2x2 list of 2x2 arrays.
    def outer_combos():
        Z = zero2()
        # S1: X=[[A00,0],[0,A11]], Y=[[B00,0],[0,B11]]
        S1 = ([blk(0,0), Z], [Z, blk(1,1)]), ([blk(0,0), Z], [Z, blk(1,1)])
        # S2: X=[[0,0],[A10,A11]], Y=[[B00,0],[0,0]]
        S2 = ([Z, Z], [blk(1,0), blk(1,1)]), ([blk(0,0), Z], [Z, Z])
        # S3: X=[[A00,0],[0,0]], Y=[[0,B01],[0,-B11]]
        S3 = ([blk(0,0), Z], [Z, Z]), ([Z, blk(0,1)], [Z, blk(1,1, -1)])
        # S4: X=[[0,0],[0,A11]], Y=[[0,0],[B10,-B00]]
        S4 = ([Z, Z], [Z, blk(1,1)]), ([Z, Z], [blk(1,0), blk(0,0, -1)])
        # S5: X=[[A00,A01],[0,0]], Y=[[0,0],[0,B11]]
        S5 = ([blk(0,0), blk(0,1)], [Z, Z]), ([Z, Z], [Z, blk(1,1)])
        # S6: X=[[-A00,0],[A10,0]], Y=[[B00,B01],[0,0]]
        S6 = ([blk(0,0, -1), Z], [blk(1,0), Z]), ([blk(0,0), blk(0,1)], [Z, Z])
        # S7: X=[[0,A01],[0,-A11]], Y=[[0,0],[B10,B11]]
        S7 = ([Z, blk(0,1)], [Z, blk(1,1, -1)]), ([Z, Z], [blk(1,0), blk(1,1)])
        return [S1, S2, S3, S4, S5, S6, S7]

    # For each outer combo (X, Y) where X, Y are 2x2 lists of 2x2 blocks,
    # expand via Strassen-2x2 into 7 inner terms.
    # Each inner term Mj = Xj * Yj where Xj, Yj are 2x2 lists of 2x2 blocks.
    # Xj assembled into 4x4 coeff matrix u, Yj into 4x4 coeff matrix v,
    # w = u @ v.
    # The C-reconstruction coefficients for M1..M7 within one S:
    # P00 = M1 - M4 + M6 - M7
    # P01 = M3 + M5
    # P10 = M2 + M4
    # P11 = M1 - M3 + M2 - M5
    # So each Mj contributes to C-blocks with specific signs.
    # The C-block (br,bc) gets Mj * sign, where the 2x2 block of w is placed
    # at C-block (br,bc) with that sign.

    def expand_outer(X, Y):
        """Return list of 7 (u, v, w_with_C_block_placement) for one outer S.
        Each term: u (4x4), v (4x4), and a list of (c_block, sign) for where
        the w = u@v 4x4 matrix goes, with sign.
        Actually w is the 4x4 product; it gets placed into C with signs per block.
        """
        # X, Y are 2x2 lists: X[0][0], X[0][1], X[1][0], X[1][1] are 2x2 arrays.
        X00, X01 = X[0][0], X[0][1]
        X10, X11 = X[1][0], X[1][1]
        Y00, Y01 = Y[0][0], Y[0][1]
        Y10, Y11 = Y[1][0], Y[1][1]

        def add2(a, b): return a + b
        def sub2(a, b): return a - b

        # 7 inner M's: (Xj, Yj)
        M = [
            (add2(X00, X11), add2(Y00, Y11)),  # M1
            (add2(X10, X11), Y00),              # M2
            (X00, sub2(Y01, Y11)),              # M3
            (X11, sub2(Y10, Y00)),              # M4
            (add2(X00, X01), Y11),              # M5
            (sub2(X10, X00), add2(Y00, Y01)),   # M6
            (sub2(X01, X11), add2(Y10, Y11)),   # M7
        ]
        # C-block placement signs for M1..M7:
        # P00 = M1 - M4 + M6 - M7  -> block(0,0)
        # P01 = M3 + M5            -> block(0,1)
        # P10 = M2 + M4            -> block(1,0)
        # P11 = M1 - M3 + M2 - M5  -> block(1,1)
        placement = {
            0: [(0, 0, 1), (1, 1, 1)],   # M1: + in C00, + in C11
            1: [(1, 0, 1), (1, 1, 1)],   # M2: + in C10, + in C11
            2: [(0, 1, 1), (1, 1, -1)],  # M3: + in C01, - in C11
            3: [(0, 0, -1), (1, 0, 1)],  # M4: - in C00, + in C10
            4: [(0, 1, 1), (1, 1, -1)],  # M5: + in C01, - in C11
            5: [(0, 0, 1)],              # M6: + in C00
            6: [(0, 0, -1)],             # M7: - in C00
        }
        terms = []
        for j in range(7):
            Xj, Yj = M[j]
            u = blockmat(Xj[0][0], Xj[0][1], Xj[1][0], Xj[1][1])
            v = blockmat(Yj[0][0], Yj[0][1], Yj[1][0], Yj[1][1])
            w = u @ v  # 4x4
            terms.append((u, v, w, placement[j]))
        return terms

    # Now build the full 4x4x4 rank-49 decomposition.
    # Each of the 7 outer combos gives 7 terms = 49 terms.
    # For each term (u, v, w, placement):
    #   u: 4x4 coeff matrix for A (flattened to 16)
    #   v: 4x4 coeff matrix for B (flattened to 16)
    #   w: 4x4 matrix, placed into C with signs per block.
    #   The w matrix is 4x4; we need to split it into 4 2x2 blocks and place
    #   each at the C-block with the given sign.
    #   But wait: w = u@v is a 4x4 matrix. The C reconstruction says
    #   C-block(br,bc) += sign * (w-block(br,bc)). So the w 4x4 matrix's
    #   block (br,bc) goes to C-block (br,bc) with sign.
    #   So for each (cbr, cbc, sign) in placement, take w[2cbr:2cbr+2, 2cbc:2cbc+2]
    #   and add sign * that to C's corresponding block in the w-vector.

    def build_strassen_4x4x4():
        """Return list of 49 (u16, v16, w16) integer vectors."""
        terms = []
        for (X, Y) in outer_combos():
            for (u, v, w, placement) in expand_outer(X, Y):
                u16 = u.flatten().tolist()
                v16 = v.flatten().tolist()
                w16 = [0] * 16
                for (cbr, cbc, sign) in placement:
                    wblock = w[2*cbr:2*cbr+2, 2*cbc:2*cbc+2]
                    for a in range(2):
                        for b in range(2):
                            idx = (2*cbr + a) * 4 + (2*cbc + b)
                            w16[idx] += sign * int(wblock[a, b])
                terms.append((u16, v16, w16))
        return terms

    # Build the rank-114 decomposition:
    # 2 Strassen 4x4x4 (rank 49 each) for B-columns 0-3 and 4-7,
    # + schoolbook for B-column 8 (4*4*1 = 16 terms).
    # Total = 49 + 49 + 16 = 114.
    strassen_terms = build_strassen_4x4x4()
    assert len(strassen_terms) == 49, f"Expected 49 Strassen terms, got {len(strassen_terms)}"

    total_rank = 49 + 49 + 16
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    def add_strassen_block(offset, b_col_offset):
        """Add 49 Strassen terms for B-columns [b_col_offset, b_col_offset+4)."""
        for t, (u16, v16, w16) in enumerate(strassen_terms):
            col = offset + t
            # u16: 16 entries for A (4x4), index (i,j) -> U[i*4+j]
            for idx, val in enumerate(u16):
                if val != 0:
                    i, j = idx // 4, idx % 4
                    U[i * m + j][col] += val
            # v16: 16 entries for B (4x4 block), index (j,k) -> B[j, b_col_offset+k]
            for idx, val in enumerate(v16):
                if val != 0:
                    j, k = idx // 4, idx % 4
                    V[j * p + (b_col_offset + k)][col] += val
            # w16: 16 entries for C (4x4), index (k,i) -> W[k*4+i]
            for idx, val in enumerate(w16):
                if val != 0:
                    k, i = idx // 4, idx % 4
                    W[k * n + i][col] += val

    add_strassen_block(0, 0)       # B columns 0-3
    add_strassen_block(49, 4)      # B columns 4-7

    # Schoolbook for B column 8: 4 (i) * 4 (j) * 1 (k) = 16 terms
    col = 98
    for i in range(n):
        for j in range(m):
            for k in [8]:
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1

    # ---------- Self-check: verify tensor decomposition on random inputs ----------
    def verify():
        rng = np.random.default_rng(42)
        A = rng.integers(-3, 4, size=(n, m)).astype(float)
        B = rng.integers(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        # Compute C from decomposition:
        # C[k*n+i] = sum_t U[i*m+j? no] ... 
        # U is (n*m, rank): U[i*m + j, t]
        # V is (m*p, rank): V[j*p + k, t]
        # W is (p*n, rank): W[k*n + i, t]
        # C[k*n + i] = sum_t sum_j U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # Wait, that's not right. The tensor decomposition:
        # T[i,j,k] = sum_t U[i,j,t] V[j,k,t] W[k,i,t]
        # where T[i,j,k] = delta (the matrix mult tensor): T[i,j,k] contributes
        # to C[i,k] += A[i,j] B[j,k]. So C[i,k] = sum_j T[i,j,k] A[i,j] B[j,k].
        # And T[i,j,k] = sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t].
        # So C[i,k] = sum_j sum_t U[i*m+j,t] V[j*p+k,t] W[k*n+i,t] A[i,j] B[j,k].
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    for t in range(total_rank):
                        s += U_arr[i * m + j, t] * V_arr[j * p + k, t] * W_arr[k * n + i, t] * A[i, j] * B[j, k]
                C_computed[i, k] = s
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
        return True

    if time.time() < deadline:
        if verify():
            return {"U": U, "V": "W" and W, "W": W}
        else:
            # Fallback to schoolbook
            return schoolbook()
    else:
        return schoolbook()
