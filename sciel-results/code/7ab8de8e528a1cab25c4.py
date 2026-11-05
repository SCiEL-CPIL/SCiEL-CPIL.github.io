"""Matrix multiplication 4,4,5 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [4, 4, 5]
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = DIMENSIONS  # 4, 4, 5

    # Strategy: Use Strassen for the 4x4x4 part (first 4 columns of B)
    # and schoolbook for the 5th column.
    #
    # The 4x4x4 part: decompose into 7 Strassen block multiplications.
    # Each block multiplication is a 2x2x2 tensor.
    # We use schoolbook (rank 8) for each 2x2x2.
    # Total: 7 * 8 = 56 terms
    # Plus 16 schoolbook terms for the 5th column.
    # Total rank = 56 + 16 = 72.

    rank = 72
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    t = 0

    # Strassen formulas for 4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Block indices:
    # A00: rows 0-1, cols 0-1
    # A01: rows 0-1, cols 2-3
    # A10: rows 2-3, cols 0-1
    # A11: rows 2-3, cols 2-3
    # B00: rows 0-1, cols 0-1
    # B01: rows 0-1, cols 2-3
    # B10: rows 2-3, cols 0-1
    # B11: rows 2-3, cols 2-3

    # Each M_i = X * Y where X and Y are 2x2 matrices that are linear combinations
    # of A and B blocks. We need to expand X*Y into rank-1 terms.

    # For M1 = (A00+A11)(B00+B11):
    # X = A00 + A11, Y = B00 + B11
    # X*Y = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Each 2x2*2x2 product has rank 8 (schoolbook), so total 4*8 = 32 terms.
    # But we can share some structure. Let's just enumerate all rank-1 terms.

    # Actually, let me think about this differently.
    # The tensor for 4x4x4 matrix multiplication has rank at most 56 using Strassen.
    # The standard Strassen decomposition for 4x4x4 uses 7 block multiplications,
    # each of which is a 2x2x2 tensor. If we use schoolbook for each 2x2x2 (rank 8),
    # we get 7*8 = 56 terms.

    # But the blocks themselves are linear combinations, so we need to expand them.
    # Let's define the 7 Strassen products and expand each into rank-1 terms.

    def add_product_term(U, V, W, t, i, j, k, a_coeff, b_coeff, c_coeff):
        """Add a rank-1 term: U[i*m+j] *= a_coeff, V[j*p+k] *= b_coeff, W[k*n+i] *= c_coeff"""
        U[i * m + j, t] += a_coeff
        V[j * p + k, t] += b_coeff
        W[k * n + i, t] += c_coeff

    # M1 = (A00+A11)(B00+B11)
    # A00: i in {0,1}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    # B00: j in {0,1}, k in {0,1}
    # B11: j in {2,3}, k in {2,3}
    #
    # (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    #
    # A00*B00: sum over i in {0,1}, j in {0,1}, k in {0,1} of A[i,j]*B[j,k]
    # A00*B11: sum over i in {0,1}, j in {2,3}, k in {2,3} of A[i,j]*B[j,k]
    # A11*B00: sum over i in {2,3}, j in {0,1}, k in {0,1} of A[i,j]*B[j,k]
    # A11*B11: sum over i in {2,3}, j in {2,3}, k in {2,3} of A[i,j]*B[j,k]

    # M1 contributes to C00 (+1) and C11 (+1)
    # C00: rows 0-1, cols 0-1
    # C11: rows 2-3, cols 2-3

    # For A00*B00: i in {0,1}, j in {0,1}, k in {0,1}
    # C[i,k] gets contribution. C00[i,k] = sum_j A[i,j]*B[j,k]
    # So W[k*n+i] gets +1 for i in {0,1}, k in {0,1}
    for i in range(2):
        for j in range(2):
            for k in range(2):
                U[i * m + j, t] += 1.0
                V[j * p + k, t] += 1.0
                W[k * n + i, t] += 1.0  # C00[i,k]
                W[k * n + i + 2, t] += 1.0  # C11[i+2,k+2] = C[i,k] for this block
                t += 1

    # For A00*B11: i in {0,1}, j in {2,3}, k in {2,3}
    # C[i,k] gets contribution. But C00[i,k] where k in {2,3} is C01[i,k-2]
    # And C11[i+2,k+2] where k in {2,3} is C11[i+2,k+2]
    # Wait, C00 is rows 0-1, cols 0-1. C01 is rows 0-1, cols 2-3.
    # C10 is rows 2-3, cols 0-1. C11 is rows 2-3, cols 2-3.
    # So C[i,k] for i in {0,1}, k in {2,3} is C01[i,k-2]
    # And C[i+2,k+2] for i in {0,1}, k in {2,3} is C11[i+2,k+2]
    for i in range(2):
        for j in range(2, 4):
            for k in range(2, 4):
                U[i * m + j, t] += 1.0
                V[j * p + k, t] += 1.0
                W[k * n + i, t] += 1.0  # C01[i,k-2]
                W[k * n + i + 2, t] += 1.0  # C11[i+2,k+2]
                t += 1

    # For A11*B00: i in {2,3}, j in {0,1}, k in {0,1}
    # C[i,k] for i in {2,3}, k in {0,1} is C10[i-2,k]
    # C[i+2,k+2] for i in {2,3}, k in {0,1} is C11[i+2,k+2]
    for i in range(2, 4):
        for j in range(2):
            for k in range(2):
                U[i * m + j, t] += 1.0
                V[j * p + k, t] += 1.0
                W[k * n + i, t] += 1.0  # C10[i-2,k]
                W[k * n + i + 2, t] += 1.0  # C11[i+2,k+2]
                t += 1

    # For A11*B11: i in {2,3}, j in {2,3}, k in {2,3}
    # C[i,k] for i in {2,3}, k in {2,3} is C11[i-2,k-2]
    # C[i+2,k+2] would be out of bounds, so only C11 gets contribution
    for i in range(2, 4):
        for j in range(2, 4):
            for k in range(2, 4):
                U[i * m + j, t] += 1.0
                V[j * p + k, t] += 1.0
                W[k * n + i, t] += 1.0  # C11[i-2,k-2]
                t += 1

    # M2 = (A10+A11)B00
    # A10: i in {2,3}, j in {0,1}
    # A11: i in {2,3}, j in {2,3}
    # B00: j in {0,1}, k in {0,1}
    #
    # (A10+A11)B00 = A10*B00 + A11*B00
    # A10*B00: i in {2,3}, j in {0,1}, k in {0,1}
    # A11*B00: i in {2,3}, j in {2,3}, k in {0,1} -- but B00 only has j in {0,1}, so this is zero!
    # Wait, A11 has columns 2-3, B00 has rows 0-1. The product A11*B00 is zero because
    # the column indices of A11 (2,3) don't match the row indices of B00 (0,1).
    # So (A10+A11)B00 = A10*B00 + 0 = A10*B00

    # M2 contributes to C10 (+1) and C11 (-1)
    # C10: rows 2-3, cols 0-1
    # C11: rows 2-3, cols 2-3

    # A10*B00: i in {2,3}, j in {0,1}, k in {0,1}
    for i in range(2, 4):
        for j in range(2):
            for k in range(2):
                U[i * m + j, t] += 1.0
                V[j * p + k, t] += 1.0
                W[k * n + i, t] += 1.0  # C10[i-2,k]
                W[k * n + i, t] += -1.0  # C11[i-2,k+2] -- wait, this is wrong
                t += 1

    # I'm making errors with the W indices. Let me restart with a cleaner approach.

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    t = 0

    # Let me use a helper to add a rank-1 term with proper W indexing.
    # W[k*n+i, t] = coefficient for C[i,k]

    def add_term(U, V, W, t, i, j, k, a_coeff, b_coeff, c_coeff):
        """Add: U[i*m+j] += a_coeff, V[j*p+k] += b_coeff, W[k*n+i] += c_coeff"""
        U[i * m + j, t] += a_coeff
        V[j * p + k, t] += b_coeff
        W[k * n + i, t] += c_coeff

    # M1 = (A00+A11)(B00+B11)
    # = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # Contributes +1 to C00 and +1 to C11

    # A00*B00: i in {0,1}, j in {0,1}, k in {0,1}
    # C[i,k] for i,k in {0,1} is C00[i,k]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                add_term(U, V, W, t, i, j, k, 1.0, 1.0, 1.0)  # C00[i,k]
                add_term(U, V, W, t, i, j, k, 0.0, 0.0, 1.0)  # C11[i+2,k+2]
                t += 1

    # Wait, this is getting messy. Let me just use the schoolbook for the whole thing.
    # Rank 80 is the seed. Let me try to get rank 72 with a correct Strassen implementation.

    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    t = 0

    # Strassen for 4x4x4:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # Each M_i is a 2x2 matrix product. We'll expand each into 8 rank-1 terms (schoolbook).

    # M1 = (A00+A11)(B00+B11)
    # A00+A11 has entries:
    #   (0,0): A[0,0]+A[2,2]
    #   (0,1): A[0,1]+A[2,3]
    #   (1,0): A[1,0]+A[3,2]
    #   (1,1): A[1,1]+A[3,3]
    # B00+B11 has entries:
    #   (0,0): B[0,0]+B[2,2]
    #   (0,1): B[0,1]+B[2,3]
    #   (1,0): B[1,0]+B[3,2]
    #   (1,1): B[1,1]+B[3,3]

    # M1[i,j] = sum_k (A00+A11)[i,k] * (B00+B11)[k,j]
    # This is a 2x2 matrix product, so we use schoolbook: 8 terms.
    # But the entries are linear combinations, so each "entry" is a sum of A entries and B entries.

    # Let me define the linear combinations explicitly.
    # For M1:
    # X = A00 + A11, Y = B00 + B11
    # X[i,k] for i,k in {0,1}:
    #   X[0,0] = A[0,0] + A[2,2]
    #   X[0,1] = A[0,1] + A[2,3]
    #   X[1,0] = A[1,0] + A[3,2]
    #   X[1,1] = A[1,1] + A[3,3]
    # Y[k,j] for k,j in {0,1}:
    #   Y[0,0] = B[0,0] + B[2,2]
    #   Y[0,1] = B[0,1] + B[2,3]
    #   Y[1,0] = B[1,0] + B[3,2]
    #   Y[1,1] = B[1,1] + B[3,3]

    # M1[i,j] = sum_k X[i,k] * Y[k,j]
    # = sum_k (sum of A entries) * (sum of B entries)
    # = sum over all combinations

    # This is getting very complex. Let me just use the schoolbook for the entire 4x4x5.
    # Rank 80.

    U, V, W, rank = schoolbook()

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W, rank

# EVOLVE-BLOCK-END
