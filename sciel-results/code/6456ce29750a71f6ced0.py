"""Program for matrix multiplication tensor decomposition 6x6x6."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s limit

    n, m, p = DIMENSIONS

    def _build_strassen_2x2_3x3():
        """
        Build a decomposition for 6x6 matrix multiplication using:
        - 2x2 block structure (blocks of size 3x3)
        - Strassen's rank-7 algorithm at the block level
        - Schoolbook (rank 27) for each 3x3 sub-multiplication
        
        Total rank: 7 * 27 = 189
        """
        rank = 7 * 27  # 189
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        # Strassen terms for 2x2 block multiplication:
        # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
        # Each block is 3x3.
        #
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6

        # For each Strassen term t, we have a "left" linear combination of A blocks
        # and a "right" linear combination of B blocks.
        # Then for the 3x3 schoolbook within each term, we iterate over (i,j,k).

        # Define the Strassen terms as (left_blocks, right_blocks) where
        # each is a list of (block_idx, coefficient)
        # block_idx: 0=A00, 1=A01, 2=A10, 3=A11 for A
        #            0=B00, 1=B01, 2=B10, 3=B11 for B

        strassen_terms = [
            # t=0: M1 = (A00+A11)*(B00+B11)
            ([ (0, 1), (3, 1) ], [ (0, 1), (3, 1) ]),
            # t=1: M2 = (A10+A11)*B00
            ([ (2, 1), (3, 1) ], [ (0, 1) ]),
            # t=2: M3 = A00*(B01-B11)
            ([ (0, 1) ], [ (1, 1), (3, -1) ]),
            # t=3: M4 = A11*(B10-B00)
            ([ (3, 1) ], [ (2, 1), (0, -1) ]),
            # t=4: M5 = (A00+A01)*B11
            ([ (0, 1), (1, 1) ], [ (3, 1) ]),
            # t=5: M6 = (A10-A00)*(B00+B01)
            ([ (2, 1), (0, -1) ], [ (0, 1), (1, 1) ]),
            # t=6: M7 = (A01-A11)*(B10+B11)
            ([ (1, 1), (3, -1) ], [ (2, 1), (3, 1) ]),
        ]

        # For each Strassen term and each 3x3 schoolbook index (i, j, k):
        # The rank-1 term has:
        #   U contribution: for each A block (br, bc) with coeff ca:
        #       row = br*3 + i, col = bc*3 + j  -> U[row, col_idx] += ca
        #   V contribution: for each B block (br, bc) with coeff cb:
        #       row = br*3 + j, col = bc*3 + k  -> V[row, col_idx] += cb
        #   W contribution: the result goes into C blocks:
        #       We need to figure out which C blocks get contributions.

        # Actually, let me think about this more carefully.
        # The Strassen term t computes M_t = L_t * R_t where L_t is a 3x3 linear combo of A blocks
        # and R_t is a 3x3 linear combo of B blocks.
        # M_t is a 3x3 matrix.
        # Then C is assembled from M_t's with certain coefficients.

        # The key insight: for a fixed Strassen term t and fixed 3x3 schoolbook indices (i,j,k),
        # the product element M_t[i,k] = sum_j L_t[i,j] * R_t[j,k]
        # But in the tensor decomposition, each rank-1 term corresponds to one (i,j,k) triple
        # in the 3x3 schoolbook, and the U, V, W entries encode the linear combinations.

        # Let me re-derive:
        # For Strassen term t, with left blocks [(ab, ca), ...] and right blocks [(bb, cb), ...]:
        # L_t = sum_{(ab, ca)} ca * A_block[ab]  (a 3x3 matrix)
        # R_t = sum_{(bb, cb)} cb * B_block[bb]  (a 3x3 matrix)
        # M_t = L_t * R_t (3x3 matrix product)
        #
        # The (i,k) element of M_t is: sum_j L_t[i,j] * R_t[j,k]
        # = sum_j [sum_{(ab,ca)} ca * A_block[ab][i,j]] * [sum_{(bb,cb)} cb * B_block[bb][j,k]]
        # = sum_{ab,bb} [sum_{(ab,ca)} ca * ...] 
        #
        # In the tensor decomposition, for each (i,j,k) in the 3x3 schoolbook:
        # U entry: for A block ab with row i, col j: coefficient ca
        # V entry: for B block bb with row j, col k: coefficient cb  
        # W entry: this contributes to C with some coefficient

        # The W coefficient depends on which C blocks receive M_t and with what sign.
        # Let me define: for Strassen term t, the contribution to C blocks:
        # c_blocks[t] = list of (c_block_idx, coeff)

        # C00 gets: M1(+1), M4(+1), M5(-1), M7(+1)
        # C01 gets: M3(+1), M5(+1)
        # C10 gets: M2(+1), M4(+1)
        # C11 gets: M1(+1), M2(-1), M3(+1), M6(+1)

        c_block_contrib = [
            # t=0 (M1): C00(+1), C11(+1)
            [ (0, 1), (3, 1) ],
            # t=1 (M2): C10(+1), C11(-1)
            [ (2, 1), (3, -1) ],
            # t=2 (M3): C00(-1), C01(+1), C11(+1)
            [ (0, -1), (1, 1), (3, 1) ],
            # t=3 (M4): C00(+1), C10(+1)
            [ (0, 1), (2, 1) ],
            # t=4 (M5): C00(-1), C01(+1)
            [ (0, -1), (1, 1) ],
            # t=5 (M6): C11(+1)
            [ (3, 1) ],
            # t=6 (M7): C00(+1), C11(+1)
            [ (0, 1), (3, 1) ],
        ]

        col_idx = 0
        for t in range(7):
            left_blocks, right_blocks = strassen_terms[t]
            c_contrib = c_block_contrib[t]

            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        # U: for each A block (ab, ca): row = ab_row*3+i, col = ab_col*3+j
                        # ab is encoded as block_row*2 + block_col
                        for ab, ca in left_blocks:
                            ab_row = ab // 2
                            ab_col = ab % 2
                            u_row = ab_row * 3 + i
                            u_col = ab_col * 3 + j
                            U[u_row, col_idx] += ca

                        # V: for each B block (bb, cb): row = bb_row*3+j, col = bb_col*3+k
                        for bb, cb in right_blocks:
                            bb_row = bb // 2
                            bb_col = bb % 2
                            v_row = bb_row * 3 + j
                            v_col = bb_col * 3 + k
                            V[v_row, col_idx] += cb

                        # W: for each C block (cb_idx, cc): row = cb_row*3+k, col = cb_col*3+i
                        for cb_idx, cc in c_contrib:
                            cb_row = cb_idx // 2
                            cb_col = cb_idx % 2
                            w_row = cb_row * 3 + k
                            w_col = cb_col * 3 + i
                            W[w_row, col_idx] += cc

                        col_idx += 1

        return U, V, W

    def _verify(U, V, W, n=6, m=6, p=6, num_trials=20):
        """Verify the decomposition by checking random matrix multiplications."""
        rank = U.shape[1]
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B

            # Compute using the decomposition:
            # C[i,k] = sum_{j,r} A[i,j] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Wait, that's not right. Let me re-derive.
            #
            # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # should equal 1 for all i,j,k (the matrix multiplication tensor).
            #
            # Then C[i,k] = sum_j A[i,j] * B[j,k] * T[i,j,k] = sum_j A[i,j] * B[j,k]
            #
            # So to verify: for random A, B, compute C via the bilinear form:
            # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            #
            # But this is O(n*m*p*rank) which for rank=189 is 6*6*6*189 = 40,392 ops per (i,k)
            # and we have 36 (i,k) pairs, so ~1.46M ops per trial. That's fine.

            C_computed = np.zeros((n, p))
            for i in range(n):
                for k in range(p):
                    s = 0.0
                    for j in range(m):
                        for r in range(rank):
                            s += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    C_computed[i, k] = s

            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    def _verify_fast(U, V, W, n=6, m=6, p=6, num_trials=10):
        """Faster verification using tensor contraction."""
        rank = U.shape[1]
        for _ in range(num_trials):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B

            # C[i,k] = sum_{j,r} A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # Let's vectorize:
            # For each r: contribution is (A * U_r) @ (V_r * B) * W_r... no.
            # Actually: sum_r W[k*n+i, r] * (sum_j A[i,j] * U[i*m+j, r] * B[j,k] * V[j*p+k, r])
            # = sum_r W[k*n+i, r] * (A * U_r_col) @ (V_r_col * B)  ... not quite.

            # Let me just do it with numpy broadcasting for speed:
            # Reshape A to (n, m, 1), B to (1, m, p)
            # U_r[i*m+j] for fixed r is a (n*m,) vector, reshape to (n, m)
            # V_r[j*p+k] for fixed r is a (m*p,) vector, reshape to (m, p)
            # W_r[k*n+i] for fixed r is a (p*n,) vector, reshape to (p, n)

            C_computed = np.zeros((n, p))
            for r in range(rank):
                Ur = U[:, r].reshape(n, m)
                Vr = V[:, r].reshape(m, p)
                Wr = W[:, r].reshape(p, n)
                # Contribution from term r: (A * Ur) @ Vr, then multiply by Wr
                # C_r[i,k] = sum_j A[i,j] * Ur[i,j] * Vr[j,k] * Wr[k,i]
                # = Wr.T * ((A * Ur) @ Vr)  ... element-wise
                C_r = (A * Ur) @ Vr
                C_computed += C_r * Wr.T

            if not np.allclose(C_computed, C_expected, atol=1e-9):
                return False
        return True

    # Build the Strassen-based decomposition
    U, V, W = _build_strassen_2x2_3x3()

    # Verify
    if _verify_fast(U, V, W):
        # Convert to lists for the output format
        U_list = U.tolist()
        V_list = V.tolist()
        W_list = W.tolist()
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U_s = np.zeros((n * m, rank), dtype=float)
        V_s = np.zeros((m * p, rank), dtype=float)
        W_s = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i * m + j, col] = 1
                    V_s[j * p + k, col] = 1
                    W_s[k * n + i, col] = 1
                    col += 1
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
# EVOLVE-BLOCK-END
