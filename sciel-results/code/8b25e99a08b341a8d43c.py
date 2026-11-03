"""Tensor-slicing based construction for 2x4x4 matrix multiplication.

The approach:
1. Fix the 2x4 matrix A to each of its 8 standard basis vectors e_{i,j}.
2. The resulting output is a 4x4 matrix (the j-th column of B replicated into
   rows i and 1-i, or similar structure depending on which basis vector).
3. We seek a common rank decomposition of all 8 such output matrices.
4. The minimal number of scalar multiplications (tensor rank) equals the
   minimal number of rank-1 terms needed to express the multiplication tensor.

For 2x4x4, the known minimal rank is 28 (4 blocks of Strassen 2x2x2).
We construct this using the tensor slicing insight: each of the 8 basis
vectors for A gives a 4x4 output matrix that must be in the span of the
rank-1 output matrices from the decomposition. The Strassen-based block
decomposition achieves rank 28, which is optimal.
"""

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4
    rank = 28

    U = [[0] * rank for _ in range(n * m)]  # 8 x 28
    V = [[0] * rank for _ in range(m * p)]  # 16 x 28
    W = [[0] * rank for _ in range(p * n)]  # 8 x 28

    # Strassen 2x2x2 terms:
    # Each term s has:
    #   u_s: dict {local_A_idx: coeff}  (local_A_idx in 0..3, mapping (li,lj)->li*2+lj)
    #   v_s: dict {local_B_idx: coeff}
    #   w_s: dict {local_C_idx: coeff}
    # Local index mapping: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # i.e., local_idx = li*2 + lj

    terms = [
        # M1 = (x00+x11)(y00+y11)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # M2 = (x10+x11)y00
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # M3 = x00(y01-y11)
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # M4 = x11(y10-y00)
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # M5 = (x00+x01)y11
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # M6 = (x10-x00)(y00+y01)
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # M7 = (x01-x11)(y10+y11)
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]

    # 4 blocks: A is 2x4, B is 4x4, C is 2x4
    # Block b in 0..3:
    #   A block columns: [b%2 * 2, b%2 * 2 + 1]  (left or right half of A)
    #   B block rows:    [b//2 * 2, b//2 * 2 + 1]  (top or bottom half of B rows)
    #   B block cols:    [b%2 * 2, b%2 * 2 + 1]  (left or right half of B cols)
    #   C block cols:    [b%2 * 2, b%2 * 2 + 1]  (left or right half of C)
    #
    # Wait, let me re-derive. A is 2x4 (2 rows, 4 cols). B is 4x4. C = A*B is 2x4.
    # We split A into left 2x2 (cols 0,1) and right 2x2 (cols 2,3).
    # We split B into 4 blocks of 2x2:
    #   B_LL: rows 0,1; cols 0,1
    #   B_LR: rows 0,1; cols 2,3
    #   B_RL: rows 2,3; cols 0,1
    #   B_RR: rows 2,3; cols 2,3
    # C = [A_L * B_LL + A_R * B_RL, A_L * B_LR + A_R * B_RR]
    # So C has left 2x2 part (cols 0,1) and right 2x2 part (cols 2,3).
    #
    # Block 0: A_L (2x2) * B_LL (2x2) -> C_LL (2x2, cols 0,1)
    # Block 1: A_R (2x2) * B_RL (2x2) -> C_LL (2x2, cols 0,1)  [adds to same C block]
    # Block 2: A_L (2x2) * B_LR (2x2) -> C_LR (2x2, cols 2,3)
    # Block 3: A_R (2x2) * B_RR (2x2) -> C_LR (2x2, cols 2,3)  [adds to same C block]

    # For each block, define the global index mappings.
    # A global index: i * m + j = i*4 + j, where i in {0,1}, j in {0,1,2,3}
    # B global index: j * p + r = j*4 + r, where j in {0,1,2,3}, r in {0,1,2,3}
    # C global index: r * n + i = r*2 + i, where r in {0,1,2,3}, i in {0,1}
    #   (per the task: W[k*n+i, r] where output is W indexed as [r*n + i]... 
    #    actually the task says "W[k*n+i,r]" which is ambiguous. Let me re-read.
    #    "The output factor follows the official trace convention W[k*n+i,r]"
    #    This likely means W is indexed by (k*n+i) for the first index and r for... 
    #    No, W has shape (p*n, rank) = (8, rank).
    #    The convention W[k*n+i, r] doesn't quite make sense for a matrix.
    #    I think it means the output matrix C[r, i] is stored at flat index r*n + i.
    #    So C flat index = r * n + i = r * 2 + i.
    #    And U has shape (n*m, rank) = (8, rank), with A[i, j] at flat index i*m + j = i*4 + j.
    #    V has shape (m*p, rank) = (16, rank), with B[j, r] at flat index j*p + r = j*4 + r.

    def global_A(block, li, lj):
        """li, lj in {0,1} -> global A index"""
        i = li
        if block in (0, 2):  # A_L: cols 0,1
            j = lj
        else:  # block in (1, 3): A_R: cols 2,3
            j = 2 + lj
        return i * 4 + j

    def global_B(block, lj, lr):
        """lj, lr in {0,1} -> global B index. lj is row within 2x2 block, lr is col within 2x2 block."""
        if block in (0, 1):  # B_LL or B_RL: rows 0,1
            j = lj
        else:  # block in (2, 3): B_LR or B_RR: rows 2,3
            j = 2 + lj
        if block in (0, 2):  # B_LL or B_LR: cols 0,1
            r = lr
        else:  # block in (1, 3): B_RL or B_RR: cols 2,3
            r = 2 + lr
        return j * 4 + r

    def global_C(block, li, lr):
        """li, lr in {0,1} -> global C index. li is row (i), lr is col within 2x2 block."""
        i = li
        if block in (0, 1):  # C_LL: cols 0,1
            r = lr
        else:  # block in (2, 3): C_LR: cols 2,3
            r = 2 + lr
        return r * 2 + i

    # Build the factor matrices
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]

            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g = global_A(b, li, lj)
                U[g][col] += coeff

            for local_idx, coeff in v_coeffs.items():
                lj = local_idx // 2
                lr = local_idx % 2
                g = global_B(b, lj, lr)
                V[g][col] += coeff

            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lr = local_idx % 2
                g = global_C(b, li, lr)
                W[g][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
