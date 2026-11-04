"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    # 3x3 by 3x4 matrix multiplication: output C[i,k] = sum_j A[i,j]*B[j,k]
    # U shape (9, rank), V shape (12, rank), W shape (12, rank)
    # W index: k*n + i = k*3 + i
    #
    # Partition:
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B1], [B2]] where B1 is 2x4, B2 is 1x4
    #
    # C = A11*B1 + A12*B2 (top 2 rows)
    #     A21*B1 + A22*B2 (bottom row)
    #
    # Use Strassen (rank 7) for A11*B1 (2x2 * 2x4):
    #   Split B1 = [B11, B12] where B11 is 2x2, B12 is 2x2
    #   C11 = M1 - M2, C12 = M2 + M3, where:
    #   M1 = A11 * (B11 + B12)  -> but B1 is 2x4, so we do Strassen per 2x2 block of B
    #
    # Actually, for 2x2 * 2x4, we can use Strassen on each 2x2 column-block:
    #   B1 = [B11 | B12] (each 2x2)
    #   C11 (2x2) = Strassen(A11, B11)  -> 7 terms
    #   C12 (2x2) = Strassen(A11, B12)  -> 7 terms
    #   Total: 14 terms for A11*B1
    #
    # A12*B2: A12 is 2x1, B2 is 1x4 -> 8 terms (schoolbook)
    # A21*B1: A21 is 1x2, B1 is 2x4 -> 8 terms (schoolbook)
    # A22*B2: A22 is 1x1, B2 is 1x4 -> 4 terms (schoolbook)
    #
    # Total rank: 14 + 8 + 8 + 4 = 34

    n, m, p = 3, 3, 4
    rank = 34

    U = [[0] * rank for _ in range(n * m)]  # 9 rows
    V = [[0] * rank for _ in range(m * p)]  # 12 rows
    W = [[0] * rank for _ in range(p * n)]  # 12 rows

    # A indices: U[i*m+j] = U[i*3+j]
    # B indices: V[j*p+k] = V[j*4+k]
    # C indices: W[k*n+i] = W[k*3+i]

    # Strassen for 2x2 * 2x2:
    # A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b10)
    # M4 = a11*(b10+b11)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b10)
    # M7 = (a01-a11)*(b01+b11)
    # C00 = M1 + M4 - M2 - M7
    # C01 = M3 + M5
    # C10 = M2 + M6
    # C11 = M1 - M3 + M5 + M7

    col = 0

    # Strassen for A11 * B11 (2x2 * 2x2) -> C top-left 2x2
    # A11: A[0,0], A[0,1], A[1,0], A[1,1] -> U[0], U[1], U[3], U[4]
    # B11: B[0,0], B[0,1], B[1,0], B[1,1] -> V[0], V[1], V[4], V[5]
    # C: C[0,0], C[0,1], C[1,0], C[1,1] -> W[0], W[1], W[3], W[4]

    # M1 = (a00+a11)*(b00+b11)
    U[0][col] = 1; U[4][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1

    # M2 = (a10+a11)*b00
    U[3][col] = 1; U[4][col] = 1
    V[0][col] = 1
    W[0][col] = -1; W[3][col] = 1
    col += 1

    # M3 = a00*(b01-b10)
    U[0][col] = 1
    V[1][col] = 1; V[4][col] = -1
    W[1][col] = 1; W[4][col] = -1
    col += 1

    # M4 = a11*(b10+b11)
    U[4][col] = 1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = 1
    col += 1

    # M5 = (a00+a01)*b11
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[1][col] = 1; W[4][col] = 1
    col += 1

    # M6 = (a10-a00)*(b00+b10)
    U[3][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[4][col] = 1
    W[3][col] = 1
    col += 1

    # M7 = (a01-a11)*(b01+b11)
    U[1][col] = 1; U[4][col] = -1
    V[1][col] = 1; V[5][col] = 1
    W[0][col] = -1; W[4][col] = 1
    col += 1

    # Strassen for A11 * B12 (2x2 * 2x2) -> C top-right 2x2
    # B12: B[0,2], B[0,3], B[1,2], B[1,3] -> V[2], V[3], V[6], V[7]
    # C: C[0,2], C[0,3], C[1,2], C[1,3] -> W[6], W[7], W[9], W[10]

    # M1 = (a00+a11)*(b00+b11)
    U[0][col] = 1; U[4][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1

    # M2 = (a10+a11)*b00
    U[3][col] = 1; U[4][col] = 1
    V[2][col] = 1
    W[6][col] = -1; W[9][col] = 1
    col += 1

    # M3 = a00*(b01-b10)
    U[0][col] = 1
    V[3][col] = 1; V[6][col] = -1
    W[7][col] = 1; W[10][col] = -1
    col += 1

    # M4 = a11*(b10+b11)
    U[4][col] = 1
    V[6][col] = 1; V[7][col] = 1
    W[6][col] = 1; W[10][col] = 1
    col += 1

    # M5 = (a00+a01)*b11
    U[0][col] = 1; U[1][col] = 1
    V[7][col] = 1
    W[7][col] = 1; W[10][col] = 1
    col += 1

    # M6 = (a10-a00)*(b00+b10)
    U[3][col] = 1; U[0][col] = -1
    V[2][col] = 1; V[6][col] = 1
    W[9][col] = 1
    col += 1

    # M7 = (a01-a11)*(b01+b11)
    U[1][col] = 1; U[4][col] = -1
    V[3][col] = 1; V[7][col] = 1
    W[6][col] = -1; W[10][col] = 1
    col += 1

    # A12 * B2: A12 is 2x1 (A[0,2], A[1,2] -> U[2], U[5]), B2 is 1x4 (B[2,0..3] -> V[8..11])
    # C: C[0,k] += A[0,2]*B[2,k] -> W[k*3+0]
    #     C[1,k] += A[1,2]*B[2,k] -> W[k*3+1]
    for k in range(4):
        # C[0,k]
        U[2][col] = 1
        V[8 + k][col] = 1
        W[k * 3 + 0][col] = 1
        col += 1
        # C[1,k]
        U[5][col] = 1
        V[8 + k][col] = 1
        W[k * 3 + 1][col] = 1
        col += 1

    # A21 * B1: A21 is 1x2 (A[2,0], A[2,1] -> U[6], U[7]), B1 is 2x4 (B[0..1, 0..3] -> V[0..7])
    # C: C[2,k] += A[2,j]*B[j,k] for j=0,1
    for j in range(2):
        for k in range(4):
            U[6 + j][col] = 1
            V[j * 4 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1

    # A22 * B2: A22 is 1x1 (A[2,2] -> U[8]), B2 is 1x4 (B[2,0..3] -> V[8..11])
    # C: C[2,k] += A[2,2]*B[2,k]
    for k in range(4):
        U[8][col] = 1
        V[8 + k][col] = 1
        W[k * 3 + 2][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
