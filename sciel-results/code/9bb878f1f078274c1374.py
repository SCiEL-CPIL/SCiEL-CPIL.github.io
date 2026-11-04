"""Matrix multiplication tensor decomposition for 3x4 * 4x8."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Strategy: use Kronecker-product structure to exploit p = 8 = 2*2*2.
    # We compute a 3x4 * 4x8 multiplication by viewing B as four 4x2 blocks.
    # For each 4x2 B-block, we use a low-rank decomposition of the 3x4 * 4x2 tensor.
    #
    # Key insight: For a 3x4 * 4x2 multiplication, we can use the fact that
    # the result is 3x2. We can decompose this using Strassen-like ideas.
    #
    # For 3x4 * 4x2 -> 3x2:
    # Split the 4 middle dimension into 2+2:
    #   A1 (3x2) * B1 (2x2) -> C1 (3x2)
    #   A2 (3x2) * B2 (2x2) -> C2 (3x2)
    #   C = C1 + C2
    #
    # For 3x2 * 2x2 -> 3x2:
    # Split the 3 rows into 2+1:
    #   A_top (2x2) * B (2x2) -> C_top (2x2)
    #   A_bottom (1x2) * B (2x2) -> C_bottom (1x2)
    #   C = [C_top; C_bottom]
    #
    # For 2x2 * 2x2 -> 2x2: use Strassen (rank 7)
    # For 1x2 * 2x2 -> 1x2: schoolbook rank 4 (can't do better for this shape)
    #
    # So for each 4x2 B-block:
    #   Block (3x4 * 4x2):
    #     = (3x2 * 2x2) + (3x2 * 2x2)
    #     = [(2x2*2x2) + (1x2*2x2)] + [(2x2*2x2) + (1x2*2x2)]
    #     = 7 + 4 + 7 + 4 = 22 per 4x2 block
    #
    # Four 4x2 blocks: 4 * 22 = 88. Same as before!
    #
    # Let me think differently. Can we share structure across the 4 B-blocks?
    #
    # Alternative: View the 4x8 B matrix as 8x4 transposed and use column-wise Strassen.
    # Actually, let me try a different decomposition.
    #
    # What about using the 4x4 Strassen decomposition (rank 49) for the middle dimension?
    # 3x4 * 4x8: split m=4 into 2+2, p=8 into 4+4
    #   (3x2 * 2x4) + (3x2 * 2x4)
    # For 3x2 * 2x4 -> 3x4:
    #   Split p=4 into 2+2:
    #     (3x2 * 2x2) + (3x2 * 2x2)
    #   For 3x2 * 2x2 -> 3x2:
    #     Split n=3 into 2+1:
    #       (2x2 * 2x2) + (1x2 * 2x2)
    #       = 7 + 4 = 11
    #   So 3x2 * 2x4 = 11 + 11 = 22
    #   So 3x4 * 4x8 = 22 + 22 = 44? That seems too good.
    #
    # Wait, I need to be more careful. Let me trace through properly.
    #
    # Actually, the issue is that the "cost" of a sub-problem isn't just the rank of
    # the tensor decomposition - we need to track how the U, V, W matrices combine.
    #
    # Let me just use the known best approach: recursive Strassen on 2x2x2 blocks.
    # The current approach gives rank 88. Let me verify this is correct and try
    # to find a better partition.
    #
    # For 3x4 * 4x8:
    # n=3, m=4, p=8
    #
    # Option 1: Strassen on n: split n=3 into 2+1
    #   (2x4 * 4x8) + (1x4 * 4x8)
    #   2x4 * 4x8: split m=4 into 2+2, p=8 into 4+4
    #     4 blocks of 2x2 * 2x4
    #     2x2 * 2x4: split p=4 into 2+2
    #       2 blocks of 2x2 * 2x2 (Strassen, rank 7 each)
    #       = 14
    #     4 * 14 = 56
    #   1x4 * 4x8: schoolbook = 32
    #   Total: 56 + 32 = 88
    #
    # Option 2: Strassen on p: split p=8 into 4+4
    #   (3x4 * 4x4) + (3x4 * 4x4)
    #   3x4 * 4x4: split n=3 into 2+1
    #     (2x4 * 4x4) + (1x4 * 4x4)
    #     2x4 * 4x4: split m=4 into 2+2
    #       2 blocks of 2x2 * 2x4
    #       2x2 * 2x4: split p=4 into 2+2
    #         2 blocks of 2x2 * 2x2 = 14
    #       2 * 14 = 28
    #     1x4 * 4x4 = 16
    #     Total for 3x4 * 4x4: 28 + 16 = 44
    #   Two of these: 44 + 44 = 88
    #
    # Option 3: Strassen on m: split m=4 into 2+2
    #   (3x2 * 2x8) + (3x2 * 2x8)
    #   3x2 * 2x8: split n=3 into 2+1, p=8 into 4+4
    #     4 blocks:
    #       2x2 * 2x4: 14
    #       2x2 * 2x4: 14
    #       1x2 * 2x4: 8
    #       1x2 * 2x4: 8
    #     Total: 14+14+8+8 = 44
    #   Two of these: 44 + 44 = 88
    #
    # All options give 88. The structure is quite rigid.
    #
    # Can we do better? What about using a 4x4x4 Strassen (rank 49)?
    # 3x4 * 4x8: 
    #   If we could use a 4x4x4 block... but n=3, not 4.
    #   Pad n to 4? That would waste one row.
    #   4x4 * 4x8: split p=8 into 4+4
    #     2 blocks of 4x4 * 4x4 (rank 49 each) = 98
    #   That's worse than 88.
    #
    # What about using the fact that 8 = 2^3 and doing a 3-level Strassen?
    # That's essentially what we're already doing.
    #
    # Let me try a different approach: use the Winograd or Karatsuba-like
    # decomposition for the 4x4 * 4x8 case, then adapt for 3x4.
    #
    # Actually, let me just implement the rank 88 solution carefully and verify it.
    # The current code already does this. Let me make sure it's correct.

    # Implementation of rank 88 decomposition:
    # Structure: 
    #   i in {0,1,2}, j in {0,1,2,3}, k in {0,...,7}
    #   Group i into {0,1} and {2}
    #   Group j into {0,1} and {2,3}
    #   Group k into {0,1}, {2,3}, {4,5}, {6,7}
    #   For each (i-group, j-group, k-group):
    #     If i-group is {0,1}: use Strassen (rank 7)
    #     If i-group is {2}: use schoolbook (rank 4)
    #
    # Total rank: 2 (j-groups) * 4 (k-groups) * (7 + 4) = 2*4*11 = 88

    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen coefficients for 2x2 * 2x2 -> 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    strassen_terms = [
        # (A coeffs [a11, a12, a21, a22], B coeffs [b11, b12, b21, b22], C coeffs [c11, c12, c21, c22])
        ([1,  0,  0,  1], [1,  0,  0,  1], [1,  0,  0,  1]),  # M1
        ([0,  0,  1,  1], [1,  0,  0,  0], [0,  0, 1, -1]),  # M2
        ([1,  1,  0,  0], [0,  0, 1,  1], [0,  1, 0,  1]),  # M3
        ([0,  0,  0,  1], [0, -1, 0,  1], [1,  0,-1,  0]),  # M4
        ([1,  0,  0,  0], [1,  0,-1,  0], [-1, 1, 0,  0]),  # M5
        ([-1, 0,  1,  0], [0,  1, 0,  1], [0,  0, 0,  1]),  # M6
        ([0,  1,  0, -1], [1,  0, 1,  0], [1,  0, 0,  0]),  # M7
    ]

    col = 0

    def add_strassen_block(i0, j0, k0):
        """Add Strassen block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        for t in range(7):
            a_coeffs = strassen_terms[t][0]
            b_coeffs = strassen_terms[t][1]
            c_coeffs = strassen_terms[t][2]

            # U: A indices (i, j) -> row i*m + j
            for di in range(2):
                for dj in range(2):
                    U[(i0+di)*m + (j0+dj)][col] = a_coeffs[di*2+dj]

            # V: B indices (j, k) -> row j*p + k
            for dj in range(2):
                for dk in range(2):
                    V[(j0+dj)*p + (k0+dk)][col] = b_coeffs[dj*2+dk]

            # W: C indices (k, i) -> row k*n + i
            for dk in range(2):
                for di in range(2):
                    W[(k0+dk)*n + (i0+di)][col] = c_coeffs[di*2+dk]

            col += 1

    def add_schoolbook_block(i0, j0, k0, n_rows=1, m_cols=2, p_cols=2):
        """Add schoolbook block for A[i0:i0+n_rows, j0:j0+m_cols] * B[j0:j0+m_cols, k0:k0+p_cols] -> C[i0:i0+n_rows, k0:k0+p_cols]"""
        nonlocal col
        for di in range(n_rows):
            for dj in range(m_cols):
                for dk in range(p_cols):
                    U[(i0+di)*m + (j0+dj)][col] = 1
                    V[(j0+dj)*p + (k0+dk)][col] = 1
                    W[(k0+dk)*n + (i0+di)][col] = 1
                    col += 1

    # Iterate over j-groups and k-groups
    for j0 in [0, 2]:
        for k0 in [0, 2, 4, 6]:
            # Strassen for i={0,1}
            add_strassen_block(0, j0, k0)
            # Schoolbook for i={2}
            add_schoolbook_block(2, j0, k0, n_rows=1, m_cols=2, p_cols=2)

    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        rank_val = U_arr.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank_val):
                        s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
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
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
