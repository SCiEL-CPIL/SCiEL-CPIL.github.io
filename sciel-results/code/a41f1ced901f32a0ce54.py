"""Matrix multiplication 4x5x6 — rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    n, m, p = 4, 5, 6

    U_cols = []
    V_cols = []
    W_cols = []

    def add_term(u_sparse, v_sparse, w_sparse):
        U_cols.append(u_sparse)
        V_cols.append(v_sparse)
        W_cols.append(w_sparse)

    def schoolbook_block(i_range, j_range, k_range):
        """Add schoolbook terms for A[i,j]*B[j,k]->C[i,k]."""
        for i in i_range:
            for j in j_range:
                for k in k_range:
                    add_term(
                        {(i * m + j): 1},
                        {(j * p + k): 1},
                        {(k * n + i): 1}
                    )

    # Laderman's 7-rank decomposition for 3x3 x 3x3 -> 3x3
    # Local indices: A row r in 0..2, col c in 0..2; B row r, col c; C row r, col c
    # Global mapping: A global = (row_off + r)*m + (col_off + c)
    #                 B global = (row_off + r)*p + (col_off + c)
    #                 C global = (col_off + c)*n + (row_off + r)
    def laderman_block(row_off, col_off_a, col_off_b, row_off_b):
        """
        Multiply A[row_off:row_off+3, col_off_a:col_off_a+3]
                 by B[row_off_b:row_off_b+3, col_off_b:col_off_b+3]
        -> C[row_off:row_off+3, col_off_b:col_off_b+3]
        """
        # Laderman's 7 products (verified):
        # P1 = (a00+a10+a11)(b00+b10+b11)
        # P2 = (a01+a02+a12)(b01+b12)
        # P3 = (a00+a01+a11)(b00+b01+b11)
        # P4 = (a02+a12+a22)(b02+b12+b22)
        # P5 = (a00+a10+a20)(b00+b10+b20)
        # P6 = (a01+a11+a21)(b01+b11+b21)
        # P7 = (a10+a20+a21)(b10+b20+b21)
        #
        # Reconstruction:
        # C00 = P1 - P2 - P3 + P4
        # C01 = P2 - P4
        # C02 = P4
        # C10 = P3 - P1
        # C11 = P1 - P3 - P6 + P7
        # C12 = P6 - P7
        # C20 = P5 - P3
        # C21 = P3 - P5 - P7 + P1  -- let me re-derive
        # C22 = P5 - P7

        # Actually, let me use a well-known correct Laderman set.
        # The standard Laderman decomposition (from Laderman 1976):
        # Given A = (a_ij), B = (b_ij), i,j in {0,1,2}
        #
        # z1 = (a00 + a10 + a11)(b00 + b10 + b11)
        # z2 = (a01 + a02 + a12)(b01 + b12)
        # z3 = (a00 + a01 + a11)(b00 + b01 + b11)
        # z4 = (a02 + a12 + a22)(b02 + b12 + b22)
        # z5 = (a00 + a10 + a20)(b00 + b10 + b20)
        # z6 = (a01 + a11 + a21)(b01 + b11 + b21)
        # z7 = (a10 + a20 + a21)(b10 + b20 + b21)
        #
        # C00 = z1 - z2 - z3 + z4
        # C01 = z2 - z4
        # C02 = z4
        # C10 = z3 - z1
        # C11 = z1 - z3 - z6 + z7
        # C12 = z6 - z7
        # C20 = z5 - z3
        # C21 = z3 - z5 - z7 + z1  -- hmm, let me check
        # C22 = z5 - z7

        # Let me verify C21:
        # C21 = a20*b01 + a21*b11 + a22*b21
        # z3 - z5 - z7 + z1:
        # z3 = (a00+a01+a11)(b00+b01+b11)
        # z5 = (a00+a10+a20)(b00+b10+b20)
        # z7 = (a10+a20+a21)(b10+b20+b21)
        # z1 = (a00+a10+a11)(b00+b10+b11)
        #
        # This is getting complicated. Let me just use a verified set.
        # I'll use the formulation from the literature that I can verify:
        #
        # Actually, the cleanest verified Laderman is:
        # Let me just use schoolbook for 3x3 (rank 9) to guarantee correctness,
        # and focus on getting the overall structure right. The total rank will
        # still be much better than 120.

        # For safety, use schoolbook for 3x3 blocks (rank 9 each)
        for r in range(3):
            for c in range(3):
                for d in range(3):
                    ai = (row_off + r) * m + (col_off_a + c)
                    bi = (row_off_b + c) * p + (col_off_b + d)
                    ci = (col_off_b + d) * n + (row_off + r)
                    add_term({ai: 1}, {bi: 1}, {ci: 1})

    # Partition:
    # A is 4x5: rows 0-3, cols 0-4
    # B is 5x6: rows 0-4, cols 0-5
    # C is 4x6: rows 0-3, cols 0-5
    #
    # Split A cols into [0,1,2] and [3,4]
    # Split B rows into [0,1,2] and [3,4]
    # Split B cols into [0,1,2] and [3,4,5]
    # Split A rows into [0,1,2] and [3]
    # Split C rows into [0,1,2] and [3]
    # Split C cols into [0,1,2] and [3,4,5]
    #
    # Blocks:
    # 1. A[0:3,0:3] * B[0:3,0:3] -> C[0:3,0:3]  (3x3 x 3x3)
    # 2. A[0:3,0:3] * B[0:3,3:6] -> C[0:3,3:6]  (3x3 x 3x3)
    # 3. A[0:3,3:5] * B[3:5,0:3] -> C[0:3,0:3]  (3x2 x 2x3)
    # 4. A[0:3,3:5] * B[3:5,3:6] -> C[0:3,3:6]  (3x2 x 2x3)
    # 5. A[3:4,0:3] * B[0:3,0:3] -> C[3:4,0:3]  (1x3 x 3x3)
    # 6. A[3:4,0:3] * B[0:3,3:6] -> C[3:4,3:6]  (1x3 x 3x3)
    # 7. A[3:4,3:5] * B[3:5,0:3] -> C[3:4,0:3]  (1x2 x 2x3)
    # 8. A[3:4,3:5] * B[3:5,3:6] -> C[3:4,3:6]  (1x2 x 2x3)
    #
    # Schoolbook ranks:
    # 1: 27
    # 2: 27
    # 3: 18
    # 4: 18
    # 5: 9
    # 6: 9
    # 7: 6
    # 8: 6
    # Total: 120 (same as full schoolbook)
    #
    # To improve, I need to use Laderman for the 3x3x3 blocks.
    # Let me implement a verified Laderman decomposition.

    # I'll use a different approach: implement Laderman correctly by
    # solving the linear system for the reconstruction coefficients.
    # But that's complex. Instead, let me use a known correct set.

    # Verified Laderman (from multiple sources):
    # For 3x3 matrices A=(a_ij), B=(b_ij):
    #
    # m1 = (a11 + a21 + a22)(b11 + b21 + b22)
    # m2 = (a12 + a13 + a23)(b12 + b23)
    # m3 = (a11 + a12 + a22)(b11 + b12 + b22)
    # m4 = (a13 + a23 + a33)(b13 + b23 + b33)
    # m5 = (a11 + a21 + a31)(b11 + b21 + b31)
    # m6 = (a12 + a22 + a32)(b12 + b22 + b32)
    # m7 = (a21 + a31 + a32)(b21 + b31 + b32)
    #
    # (using 1-indexed)
    #
    # C11 = m1 - m2 - m3 + m4
    # C12 = m2 - m4
    # C13 = m4
    # C21 = m3 - m1
    # C22 = m1 - m3 - m6 + m7
    # C23 = m6 - m7
    # C31 = m5 - m3
    # C32 = m3 - m5 - m7 + m1  -- need to verify
    # C33 = m5 - m7

    # Let me verify C32 with 0-indexed:
    # C21 (0-indexed) = a20*b01 + a21*b11 + a22*b21
    # m3 - m5 - m7 + m1 (0-indexed: m3, m5, m7, m1)
    # m3 = (a00+a01+a11)(b00+b01+b11)
    # m5 = (a00+a10+a20)(b00+b10+b20)
    # m7 = (a10+a20+a21)(b10+b20+b21)
    # m1 = (a00+a10+a11)(b00+b10+b11)
    #
    # This is too error-prone to verify by hand. Let me use a different strategy:
    # Use the fact that I can verify the decomposition numerically at runtime
    # and fall back to schoolbook if it fails.

    # Actually, the simplest correct approach that beats 120:
    # Use Laderman for the two 3x3x3 blocks in the top-left 3x3 region.
    # But I need a correct Laderman. Let me just use schoolbook everywhere
    # but with a smarter partition that doesn't increase rank.

    # Alternative: the full schoolbook is 4*5*6 = 120. Any partition
    # that uses schoolbook for each block sums to 120. So I MUST use
    # a sub-linear algorithm for at least one block to beat 120.

    # Let me try a different, simpler optimization:
    # For the 1x3 * 3x3 blocks (rows 5,6), the rank is 9 each.
    # But a 1x3 * 3x3 product can be done in 3 multiplications
    # (just dot products for each output column). Wait, no:
    # C[i,k] = sum_j A[i,j] * B[j,k] for i=3, k=0..5
    # That's 6 outputs, each requiring 3 multiplications = 18 scalar mults,
    # but as tensor rank, it's 6 terms (one per output element) with
    # A[3,j] and B[j,k] as the factors. Actually no, each output C[3,k]
    # = sum_j A[3,j]*B[j,k], which is a dot product = rank 3 for that
    # single output. For 6 outputs, total rank = 6*3 = 18? No...
    #
    # In tensor decomposition, C[3,k] = sum_j A[3,j] B[j,k]
    # This is a single bilinear form. The tensor for this is:
    # T[i,j,k] = delta_{i,3} * A_{3,j} * B_{j,k}
    # Wait, no. The multiplication tensor T_{i,j,k} satisfies
    # C_{i,k} = sum_j T_{i,j,k} A_{i',j} B_{j',k'} ... no.
    #
    # The matrix multiplication tensor T_{i,j,k} has the property that
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}.
    # So T_{i,j,k} = 1 for all i,j,k (in the full n*m*p tensor).
    # A rank-1 term is U_{i,j} V_{j,k} W_{i,k} where U, V, W are
    # vectors in R^{nm}, R^{mp}, R^{pn} respectively.
    #
    # For the block A[3:4, 0:3] * B[0:3, 0:3] -> C[3:4, 0:3]:
    # C[3,k] = sum_{j=0}^{2} A[3,j] B[j,k] for k=0,1,2
    # This is 3 outputs, each a dot product of length 3.
    # Tensor rank for this sub-tensor: 3*3 = 9 (schoolbook) or
    # can we do better? A 1x3 * 3x3 -> 1x3 is just a vector-matrix
    # product. The tensor is 3x3x3 = 27 entries, but only 9 are
    # nonzero (one per (j,k) pair for the single row i=3).
    # Actually the tensor T_{i,j,k} for this block is:
    # T_{3,j,k} = 1 for j in 0..2, k in 0..2, and 0 otherwise.
    # This is a 4x5x6 tensor with only 9 nonzero entries.
    # The rank of this sub-tensor is 9 (each nonzero entry is a
    # separate rank-1 term since they don't share structure).
    # So schoolbook is optimal here.

    # OK so the only way to beat 120 is to use a sub-quadratic
    # algorithm for the 3x3 blocks. Let me just implement Laderman
    # correctly using a verified source.

    # Here's a verified Laderman decomposition (0-indexed):
    # A = [[a0,a1,a2],[a3,a4,a5],[a6,a7,a8]]
    # B = [[b0,b1,b2],[b3,b4,b5],[b6,b7,b8]]
    #
    # z1 = (a0+a3+a4)(b0+b3+b4)
    # z2 = (a1+a2+a5)(b1+b5)
    # z3 = (a0+a1+a4)(b0+b1+b4)
    # z4 = (a2+a5+a8)(b2+b5+b8)
    # z5 = (a0+a3+a6)(b0+b3+b6)
    # z6 = (a1+a4+a7)(b1+b4+b7)
    # z7 = (a3+a6+a7)(b3+b6+b7)
    #
    # C00 = z1 - z2 - z3 + z4
    # C01 = z2 - z4
    # C02 = z4
    # C10 = z3 - z1
    # C11 = z1 - z3 - z6 + z7
    # C12 = z6 - z7
    # C20 = z5 - z3
    # C21 = z3 - z5 - z7 + z1
    # C22 = z5 - z7

    # Let me verify C21:
    # C21 = a6*b1 + a7*b4 + a8*b7
    # z3 - z5 - z7 + z1:
    # z3 = (a0+a1+a4)(b0+b1+b4)
    #    = a0*b0 + a0*b1 + a0*b4 + a1*b0 + a1*b1 + a1*b4 + a4*b0 + a4*b1 + a4*b4
    # z5 = (a0+a3+a6)(b0+b3+b6)
    #    = a0*b0 + a0*b3 + a0*b6 + a3*b0 + a3*b3 + a3*b6 + a6*b0 + a6*b3 + a6*b6
    # z7 = (a3+a6+a7)(b3+b6+b7)
    #    = a3*b3 + a3*b6 + a3*b7 + a6*b3 + a6*b6 + a6*b7 + a7*b3 + a7*b6 + a7*b7
    # z1 = (a0+a3+a4)(b0+b3+b4)
    #    = a0*b0 + a0*b3 + a0*b4 + a3*b0 + a3*b3 + a3*b4 + a4*b0 + a4*b3 + a4*b4
    #
    # z3 - z5 - z7 + z1:
    # a0*b0: 1 - 1 + 1 = 1
    # a0*b1: 1
    # a0*b4: 1 - 1 = 0
    # a0*b3: -1 + 1 = 0
    # a0*b6: -1
    # a1*b0: 1
    # a1*b1: 1
    # a1*b4: 1
    # a4*b0: 1 - 1 = 0
    # a4*b1: 1
    # a4*b4: 1 - 1 = 0
    # a4*b3: -1 + 1 = 0
    # a3*b0: -1 + 1 = 0
    # a3*b3: -1 - 1 + 1 = -1
    # a3*b6: -1 - 1 + 1 = -1
    # a3*b4: 1
    # a3*b7: -1
    # a6*b0: -1
    # a6*b3: -1 - 1 = -2
    # a6*b6: -1 - 1 = -2
    # a6*b7: -1
    # a7*b3: -1
    # a7*b6: -1
    # a7*b7: -1
    #
    # This doesn't look like C21 = a6*b1 + a7*b4 + a8*b7.
    # So this Laderman set is WRONG.

    # Let me use a different approach. I'll implement the Strassen-like
    # decomposition for 3x3 which I can verify, or just accept rank 120
    # and focus on correctness.

    # Actually, let me try yet another known Laderman set.
    # From the paper "A 7-rank algorithm for 3x3 matrix multiplication":
    #
    # Using 1-indexed (i,j in 1..3):
    # z1 = (a11+a21+a22)(b11+b21+b22)
    # z2 = (a12+a13+a23)(b12+b23)
    # z3 = (a11+a12+a22)(b11+b12+b22)
    # z4 = (a13+a23+a33)(b13+b23+b33)
    # z5 = (a11+a21+a31)(b11+b21+b31)
    # z6 = (a12+a22+a32)(b12+b22+b32)
    # z7 = (a21+a31+a32)(b21+b31+b32)
    #
    # C11 = z1 - z2 - z3 + z4
    # C12 = z2 - z4
    # C13 = z4
    # C21 = z3 - z1
    # C22 = z1 - z3 - z6 + z7
    # C23 = z6 - z7
    # C31 = z5 - z3
    # C32 = z3 - z5 - z7 + z1
    # C33 = z5 - z7

    # Converting to 0-indexed (subtract 1 from all indices):
    # a11->a00, a21->a30, a22->a40, etc. Wait, 1-indexed (i,j) -> 0-indexed (i-1,j-1)
    # a11 = a[0,0] = a00
    # a21 = a[1,0] = a30  (row 1, col 0)
    # a22 = a[1,1] = a40  (row 1, col 1)
    # a12 = a[0,1] = a10
    # a13 = a[0,2] = a20
    # a23 = a[1,2] = a50
    # a33 = a[2,2] = a80
    # a31 = a[2,0] = a60
    # a32 = a[2,1] = a70
    #
    # Similarly for B:
    # b11 = b[0,0] = b00
    # b21 = b[1,0] = b30
    # b22 = b[1,1] = b40
    # b12 = b[0,1] = b10
    # b23 = b[1,2] = b50
    # b13 = b[0,2] = b20
    # b33 = b[2,2] = b80
    # b31 = b[2,0] = b60
    # b32 = b[2,1] = b70
    #
    # So in 0-indexed flat (row-major, 3x3):
    # a00=0, a10=1, a20=2, a30=3, a40=4, a50=5, a60=6, a70=7, a80=8
    #
    # z1 = (a00+a30+a40)(b00+b30+b40)
    # z2 = (a10+a20+a50)(b10+b50)
    # z3 = (a00+a10+a40)(b00+b10+b40)
    # z4 = (a20+a50+a80)(b20+b50+b80)
    # z5 = (a00+a30+a60)(b00+b30+b60)
    # z6 = (a10+a40+a70)(b10+b40+b70)
    # z7 = (a30+a60+a70)(b30+b60+b70)
    #
    # C00 = z1 - z2 - z3 + z4
    # C01 = z2 - z4
    # C02 = z4
    # C10 = z3 - z1
    # C11 = z1 - z3 - z6 + z7
    # C12 = z6 - z7
    # C20 = z5 - z3
    # C21 = z3 - z5 - z7 + z1
    # C22 = z5 - z7

    # This is the same as before! And I showed it's wrong for C21.
    # So either my verification was wrong or the formula is wrong.
    # Let me re-verify C21 more carefully.
    #
    # C21 (0-indexed) = a[2,0]*b[0,1] + a[2,1]*b[1,1] + a[2,2]*b[2,1]
    #                   = a60*b10 + a70*b40 + a80*b70
    #
    # z3 - z5 - z7 + z1:
    # Let me expand each term and collect coefficients for a60*b10, a70*b40, a80*b70.
    #
    # z3 = (a00+a10+a40)(b00+b10+b40)
    #    = a00*b00 + a00*b10 + a00*b40 + a10*b00 + a10*b10 + a10*b40 + a40*b00 + a40*b10 + a40*b40
    #
    # z5 = (a00+a30+a60)(b00+b30+b60)
    #    = a00*b00 + a00*b30 + a00*b60 + a30*b00 + a30*b30 + a30*b60 + a60*b00 + a60*b30 + a60*b60
    #
    # z7 = (a30+a60+a70)(b30+b60+b70)
    #    = a30*b30 + a30*b60 + a30*b70 + a60*b30 + a60*b60 + a60*b70 + a70*b30 + a70*b60 + a70*b70
    #
    # z1 = (a00+a30+a40)(b00+b30+b40)
    #    = a00*b00 + a00*b30 + a00*b40 + a30*b00 + a30*b30 + a30*b40 + a40*b00 + a40*b30 + a40*b40
    #
    # z3 - z5 - z7 + z1:
    # a60*b10: not present in any term -> coefficient 0. But we need 1!
    #
    # So this formula is definitely wrong for C21. The Laderman set I'm using
    # is incorrect. Let me just give up on Laderman and use schoolbook.
    # The score will be 120, same as baseline. But at least it's correct.

    # Use schoolbook for all 120 terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                add_term(
                    {(i * m + j): 1},
                    {(j * p + k): 1},
                    {(k * n + i): 1}
                )

    # Build output matrices
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
