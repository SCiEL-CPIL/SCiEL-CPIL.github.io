"""Program for rank-minimal exact tensor decomposition of 2x3x8 matrix multiplication."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: shape (n*m, rank), V: shape (m*p, rank), W: shape (p*n, rank)
    # Index convention:
    #   U[i*m + j, c]
    #   V[j*p + k, c]
    #   W[k*n + i, c]
    # The tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] must equal 1
    # for all valid i,j,k (since (A*B)[i,k] = sum_j A[i,j]*B[j,k]).

    # We decompose into two parts:
    # Part 1: j in {0,1} -> 2x2x8 tensor
    # Part 2: j = 2 -> 2x1x8 tensor

    # Part 2: Schoolbook for j=2, k in 0..7
    # 8 multiplications: U[i*3+2, c]=1, V[2*8+k, c]=1, W[k*2+i, c]=1
    part2_rank = 8
    part2_cols = list(range(part2_rank))

    # Part 1: 2x2x8 using sliding window Strassen
    # Split 8 columns into 4 blocks of size 2.
    # For each block of 2 k-columns, use Strassen's 7 multiplications.
    # But we can share: the 7 Strassen products for block t use combinations of
    # A columns (i, j) for j in {0,1} and B rows (j, k) for j in {0,1}, k in block.
    #
    # Standard Strassen for 2x2x2:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B00)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    # where Aij = A[i,j], Bjk = B[j,k]
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # For each block t (k in {2t, 2t+1}), we have 7 multiplications.
    # Total for Part 1: 4 blocks * 7 = 28 multiplications.
    part1_rank = 28
    part1_cols = list(range(part1_rank, part1_rank + part2_rank))

    total_rank = part1_rank + part2_rank

    U = [[0] * total_rank for _ in range(n * m)]  # 6 rows
    V = [[0] * total_rank for _ in range(m * p)]  # 24 rows
    W = [[0] * total_rank for _ in range(p * n)]  # 16 rows

    # Fill Part 2 (schoolbook for j=2)
    for c_idx, col in enumerate(part2_cols):
        for i in range(n):
            U[i * m + 2][col] = 1
        for k in range(p):
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
    # Wait, the above loop for W has a bug: i is not in scope after the first loop.
    # Let me redo Part 2 properly.

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    # Part 2: j=2, schoolbook
    col_offset = 0
    for k in range(p):
        for i in range(n):
            U[i * m + 2][col_offset] = 1
            V[2 * p + k][col_offset] = 1
            W[k * n + i][col_offset] = 1
            col_offset += 1
    # col_offset is now 8 = part2_rank

    # Part 1: j in {0,1}, sliding window Strassen for 4 blocks of 2 k-columns
    col_offset = part2_rank
    for t in range(4):
        k0 = 2 * t
        k1 = 2 * t + 1
        # 7 Strassen multiplications
        # M1 = (A[0,0]+A[1,1]) * (B[0,k0]+B[1,k1])
        # M2 = (A[1,0]+A[1,1]) * B[0,k0]
        # M3 = A[0,0] * (B[1,k0]-B[1,k1])
        # M4 = A[1,1] * (B[0,k1]-B[0,k0])
        # M5 = (A[0,0]+A[0,1]) * B[1,k1]
        # M6 = (A[1,0]-A[0,0]) * (B[0,k0]+B[1,k0])
        # M7 = (A[0,1]-A[1,1]) * (B[0,k1]+B[1,k1])
        
        # M1
        c = col_offset
        U[0 * m + 0][c] = 1  # A[0,0]
        U[1 * m + 1][c] = 1  # A[1,1]
        V[0 * p + k0][c] = 1  # B[0,k0]
        V[1 * p + k1][c] = 1  # B[1,k1]
        # W: C[0,k0] += M1, C[0,k1] -= M1, C[1,k0] -= M1, C[1,k1] -= M1
        W[k0 * n + 0][c] = 1
        W[k0 * n + 1][c] = -1
        W[k1 * n + 0][c] = -1
        W[k1 * n + 1][c] = -1
        col_offset += 1

        # M2
        c = col_offset
        U[1 * m + 0][c] = 1  # A[1,0]
        U[1 * m + 1][c] = 1  # A[1,1]
        V[0 * p + k0][c] = 1  # B[0,k0]
        # W: C[1,k0] += M2, C[1,k1] -= M2
        W[k0 * n + 1][c] = 1
        W[k1 * n + 1][c] = -1
        col_offset += 1

        # M3
        c = col_offset
        U[0 * m + 0][c] = 1  # A[0,0]
        V[1 * p + k0][c] = 1  # B[1,k0]
        V[1 * p + k1][c] = -1  # -B[1,k1]
        # W: C[0,k1] += M3, C[1,k0] += M3
        W[k1 * n + 0][c] = 1
        W[k0 * n + 1][c] = 1
        col_offset += 1

        # M4
        c = col_offset
        U[1 * m + 1][c] = 1  # A[1,1]
        V[0 * p + k1][c] = 1  # B[0,k1]
        V[0 * p + k0][c] = -1  # -B[0,k0]
        # W: C[0,k0] += M4, C[1,k0] += M4
        W[k0 * n + 0][c] = 1
        W[k0 * n + 1][c] = 1
        col_offset += 1

        # M5
        c = col_offset
        U[0 * m + 0][c] = 1  # A[0,0]
        U[0 * m + 1][c] = 1  # A[0,1]
        V[1 * p + k1][c] = 1  # B[1,k1]
        # W: C[0,k0] -= M5, C[0,k1] += M5
        W[k0 * n + 0][c] = -1
        W[k1 * n + 0][c] = 1
        col_offset += 1

        # M6
        c = col_offset
        U[1 * m + 0][c] = 1  # A[1,0]
        U[0 * m + 0][c] = -1  # -A[0,0]
        V[0 * p + k0][c] = 1  # B[0,k0]
        V[1 * p + k0][c] = 1  # B[1,k0]
        # W: C[1,k0] += M6
        W[k0 * n + 1][c] = 1
        col_offset += 1

        # M7
        c = col_offset
        U[0 * m + 1][c] = 1  # A[0,1]
        U[1 * m + 1][c] = -1  # -A[1,1]
        V[0 * p + k1][c] = 1  # B[0,k1]
        V[1 * p + k1][c] = 1  # B[1,k1]
        # W: C[0,k1] -= M7, C[1,k1] += M7
        W[k1 * n + 0][c] = -1
        W[k1 * n + 1][c] = 1
        col_offset += 1

    assert col_offset == total_rank, f"col_offset={col_offset}, total_rank={total_rank}"

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
