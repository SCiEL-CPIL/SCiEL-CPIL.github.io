"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Strategy: Block-Strassen decomposition using 2x2 Strassen on 3x3 blocks.
    # 6x6 = 2x2 blocks of 3x3.
    # Strassen rank for 2x2 is 7 instead of 4, giving rank 7 * 3 * 3 * 3 = 189.
    # Then apply SVD-based rank reduction to remove linearly dependent columns.

    # Strassen 2x2 multiplication:
    # C = A * B where A, B are 2x2 matrices
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 6x6 with 3x3 blocks:
    # A = [[A11, A12], [A21, A22]] where each is 3x3
    # B = [[B11, B12], [B21, B22]] where each is 3x3
    # C = A * B

    # Each Mj is a product of 3x3 matrices, requiring 9 scalar multiplications.
    # Total rank = 7 * 9 = 63... wait, no.
    # Actually, each "multiplication" Mj = X * Y where X and Y are 3x3 matrices.
    # The tensor for 3x3 * 3x3 has rank 27 (schoolbook).
    # So total rank = 7 * 27 = 189.

    # Let's build the factor matrices.
    # U has shape (n*m, rank) = (36, 189)
    # V has shape (m*p, rank) = (36, 189)
    # W has shape (p*n, rank) = (36, 189)

    # For each Strassen step j (0..6), and for each (i,k) in 3x3 block:
    # The contribution to U, V, W.

    # Let's define the Strassen factors for 2x2:
    # M1: (A11+A22)*(B11+B22)
    # M2: (A21+A22)*B11
    # M3: A11*(B12-B22)
    # M4: A22*(B21-B11)
    # M5: (A11+A12)*B22
    # M6: (A21-A11)*(B11+B12)
    # M7: (A12-A22)*(B21+B22)

    # In terms of block indices:
    # A11 = rows 0-2, cols 0-2 of A
    # A12 = rows 0-2, cols 3-5 of A
    # A21 = rows 3-5, cols 0-2 of A
    # A22 = rows 3-5, cols 3-5 of A

    # Similarly for B and C.

    # For each Strassen multiplication, we have a linear combination of A blocks
    # multiplied by a linear combination of B blocks.

    # Let's define the coefficients:
    # For each j, we have:
    # X_j = sum_i alpha_{j,i} * A_i  (where A_i are the 4 blocks)
    # Y_j = sum_k beta_{j,k} * B_k  (where B_k are the 4 blocks)
    # M_j = X_j * Y_j (3x3 * 3x3)

    # Then C is assembled from M_j.

    # Strassen coefficients:
    # j=0 (M1): X = A11 + A22, Y = B11 + B22
    # j=1 (M2): X = A21 + A22, Y = B11
    # j=2 (M3): X = A11, Y = B12 - B22
    # j=3 (M4): X = A22, Y = B21 - B11
    # j=4 (M5): X = A11 + A12, Y = B22
    # j=5 (M6): X = A21 - A11, Y = B11 + B12
    # j=6 (M7): X = A12 - A22, Y = B21 + B22

    # Block index mapping: 0=A11/B11, 1=A12/B12, 2=A21/B21, 3=A22/B22
    # alpha[j][block] and beta[j][block]

    alpha = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1], # M7
    ]
    beta = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [0, 0, 1, -1], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1],  # M7
    ]

    # C assembly:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # gamma[c_block][j] where c_block is 0=C11, 1=C12, 2=C21, 3=C22
    gamma = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0],   # C22
    ]

    # Now build U, V, W.
    # For each j (0..6) and each (a, b) in 3x3 (the inner product indices):
    # The column index is j * 9 + a * 3 + b

    rank = 7 * 9
    U = np.zeros((n * m, rank))  # (36, 189)
    V = np.zeros((m * p, rank))  # (36, 189)
    W = np.zeros((p * n, rank))  # (36, 189)

    for j in range(7):
        for a in range(3):  # row index within 3x3 block
            for b in range(3):  # col index within 3x3 block
                col = j * 9 + a * 3 + b

                # U contribution: for each block i where alpha[j][i] != 0
                # A_i is a 3x3 block. The element A_i[a, b] contributes to U.
                # A_i row offset: i // 2 * 3, col offset: i % 2 * 3
                for i in range(4):
                    coeff = alpha[j][i]
                    if coeff != 0:
                        row_offset = (i // 2) * 3
                        col_offset = (i % 2) * 3
                        U[row_offset + a, col_offset + b] += coeff

                # V contribution: for each block k where beta[j][k] != 0
                # B_k is a 3x3 block. The element B_k[a, b] contributes to V.
                # B_k row offset: k // 2 * 3, col offset: k % 2 * 3
                for k in range(4):
                    coeff = beta[j][k]
                    if coeff != 0:
                        row_offset = (k // 2) * 3
                        col_offset = (k % 2) * 3
                        V[row_offset + a, col_offset + b] += coeff

                # W contribution: for each block c where gamma[c][j] != 0
                # C_c is a 3x3 block. The element C_c[a, b] gets contribution.
                # W index: k * n + i where k is col of C, i is row of C
                # C_c row offset: c // 2 * 3, col offset: c % 2 * 3
                for c in range(4):
                    coeff = gamma[c][j]
                    if coeff != 0:
                        row_offset = (c % 2) * 3  # col of C block -> k in W[k*n+i]
                        col_offset = (c // 2) * 3  # row of C block -> i in W[k*n+i]
                        W[row_offset * n + col_offset + a, col] += coeff * 1  # Note: W[k*n+i, col]

    # Wait, I need to be careful with the W indexing.
    # W[k*n+i, col] where k is the column index in C (0..5) and i is the row index in C (0..5).
    # For block c: rows are c//2*3 to c//2*3+2, cols are c%2*3 to c%2*3+2.
    # So for element C_c[a, b]: row = c//2*3 + a, col = c%2*3 + b.
    # W index = col * n + row = (c%2*3 + b) * 6 + (c//2*3 + a)

    # Let me redo W:
    W = np.zeros((p * n, rank))
    for j in range(7):
        for a in range(3):
            for b in range(3):
                col = j * 9 + a * 3 + b
                for c in range(4):
                    coeff = gamma[c][j]
                    if coeff != 0:
                        row_in_C = (c // 2) * 3 + a
                        col_in_C = (c % 2) * 3 + b
                        W[col_in_C * n + row_in_C, col] += coeff

    # Now apply SVD-based rank reduction.
    # The idea: if columns of U, V, W are linearly dependent, we can reduce rank.
    # We need to find a smaller set of columns that still spans the same tensor.

    # Actually, for exact decomposition, we need to be more careful.
    # Let's try to find the numerical rank of the factor matrices.

    if time.time() < deadline - 10:
        # Try to reduce rank by finding linear dependencies.
        # Compute SVD of U to find its numerical rank.
        try:
            # Use a tolerance for numerical rank
            tol = 1e-10

            # Compute ranks
            U_rank = np.linalg.matrix_rank(U, tol=tol)
            V_rank = np.linalg.matrix_rank(V, tol=tol)
            W_rank = np.linalg.matrix_rank(W, tol=tol)

            # The effective rank is min of these, but we need to be careful.
            # For a valid decomposition, all three must have the same effective rank
            # in the sense that the tensor is exactly represented.

            # Let's try a different approach: use the fact that if a column of U is
            # a linear combination of other columns, we can eliminate it.

            # Actually, let's just return the Block-Strassen result for now,
            # as the SVD reduction is complex and may not preserve exactness.

            # But let's at least check if we can do better with a simple approach.
            # For now, return the 189-rank decomposition.

        except Exception:
            pass

    # Convert to lists of Fractions for exactness
    def to_fraction_list(matrix):
        result = []
        for row in matrix:
            row_list = []
            for val in row:
                if abs(val - round(val)) < 1e-10:
                    row_list.append(Fraction(int(round(val))))
                else:
                    row_list.append(Fraction(val).limit_denominator(1000000))
            result.append(row_list)
        return result

    U_frac = to_fraction_list(U)
    V_frac = to_fraction_list(V)
    W_frac = to_fraction_list(W)

    return {"U": U_frac, "V": V_frac, "W": W_frac}
# EVOLVE-BLOCK-END
