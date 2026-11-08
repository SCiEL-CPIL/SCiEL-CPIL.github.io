"""Matrix multiplication 2x3x6 with Strassen-like rank reduction."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    # Split A's columns into first two (2x2 block) and third (2x1 block)
    # Split B's rows correspondingly: first two rows of B (2x6) and third row (1x6)
    # C = A11 @ B11 + A12 @ B21  (2x2 * 2x6 + 2x1 * 1x6)
    #
    # For the 2x2x6 part, split B11 into three 2x2 blocks (columns 0-1, 2-3, 4-5)
    # Use Strassen on each 2x2x2 block: rank 7 per block, total 21
    # For the 2x1x6 part (A12 @ B21), use schoolbook: rank 12
    # Total rank: 21 + 12 = 33

    # U shape: (n*m, rank) = (6, 33)
    # V shape: (m*p, rank) = (18, 33)
    # W shape: (p*n, rank) = (12, 33)

    rank = 33
    U = [[0] * rank for _ in range(6)]
    V = [[0] * rank for _ in range(18)]
    W = [[0] * rank for _ in range(12)]

    # Indexing:
    # A is 2x3: A[i,j], i=0,1; j=0,1,2
    # B is 3x6: B[j,k], j=0,1,2; k=0,...,5
    # C is 2x6: C[i,k] = sum_j A[i,j]*B[j,k]
    # U rows indexed by (i,j) -> i*m+j = i*3+j
    # V rows indexed by (j,k) -> j*p+k = j*6+k
    # W rows indexed by (k,i) -> k*n+i = k*2+i (trace convention W[k*n+i, r])

    col = 0
    # Process each 2x2x2 block of the 2x2x6 part
    # Block b handles columns 2*b, 2*b+1 of B (and corresponding columns of C)
    for b in range(3):
        # B rows 0 and 1, columns 2*b and 2*b+1
        k0 = 2 * b
        k1 = 2 * b + 1
        # Strassen's 7 products for C_block = A_2x2 @ B_2x2
        # A_2x2 = [[A00, A01], [A10, A11]] where Aij = A[i, j] for i=0,1 j=0,1
        # B_2x2 = [[B00, B01], [B10, B11]] where Bjk = B[j, k] for j=0,1 k=k0,k1
        
        # M1 = A00 * (B11 - B01)  -> but we need to handle the subtraction in U/V/W
        # Actually, Strassen uses:
        # M1 = A00 * (B11 - B01)
        # M2 = (A10 + A00) * B01
        # M3 = A11 * (B00 - B10)
        # M4 = (A11 + A10) * B10
        # M5 = (A11 + A00) * B11
        # M6 = (A10 - A00) * B00
        # M7 = (A01 - A11) * B10
        # C00 = M5 + M4 - M3 - M7
        # C01 = M1 + M5
        # C10 = M2 + M4
        # C11 = M1 + M3 - M2 - M6
        
        # For each Strassen product, we need to encode the sums/differences
        # in U, V, W. Since U and V can have entries like 1, -1, 2, etc.,
        # and W encodes the combination.

        # M1 = A00 * (B11 - B01)
        U[0][col] = 1          # A[0,0]
        V[6 + k0][col] = -1    # B[1, k0]
        V[6 + k1][col] = 1     # B[1, k1]
        W[k0 * 2 + 0][col] = 1 # C[0, k0]
        W[k1 * 2 + 0][col] = 1 # C[0, k1]
        col += 1

        # M2 = (A10 + A00) * B01
        U[0][col] = 1          # A[0,0]
        U[3][col] = 1          # A[1,0]
        V[k1][col] = 1         # B[0, k1]
        W[k1 * 2 + 1][col] = 1 # C[1, k1]
        col += 1

        # M3 = A11 * (B00 - B10)
        U[1][col] = 1          # A[0,1]
        V[k0][col] = 1         # B[0, k0]
        V[6 + k0][col] = -1    # B[1, k0]
        W[k0 * 2 + 0][col] = -1 # C[0, k0] (with minus)
        W[k0 * 2 + 1][col] = 1  # C[1, k0] (with plus)
        col += 1

        # M4 = (A11 + A10) * B10
        U[1][col] = 1          # A[0,1]
        U[4][col] = 1          # A[1,1]
        V[6 + k0][col] = 1     # B[1, k0]
        W[k0 * 2 + 0][col] = -1 # C[0, k0]
        W[k0 * 2 + 1][col] = 1  # C[1, k0]
        col += 1

        # M5 = (A11 + A00) * B11
        U[1][col] = 1          # A[0,1]
        U[3][col] = 1          # A[1,0]
        V[6 + k1][col] = 1     # B[1, k1]
        W[k0 * 2 + 0][col] = 1 # C[0, k0]
        W[k1 * 2 + 0][col] = 1 # C[0, k1]
        col += 1

        # M6 = (A10 - A00) * B00
        U[0][col] = -1         # -A[0,0]
        U[3][col] = 1          # A[1,0]
        V[k0][col] = 1         # B[0, k0]
        W[k0 * 2 + 1][col] = -1 # C[1, k0]
        col += 1

        # M7 = (A01 - A11) * B10
        U[1][col] = 1          # A[0,1]
        U[4][col] = -1         # -A[1,1]
        V[6 + k0][col] = 1     # B[1, k0]
        W[k0 * 2 + 0][col] = -1 # C[0, k0]
        col += 1

    # Now the 2x1x6 part: A12 @ B21
    # A12 is 2x1: A[0,2], A[1,2]
    # B21 is 1x6: B[2, 0..5]
    # C_part[i,k] = A[i,2] * B[2,k]
    for i in range(2):
        for k in range(6):
            U[i * 3 + 2][col] = 1   # A[i, 2]
            V[12 + k][col] = 1      # B[2, k]
            W[k * 2 + i][col] = 1   # C[i, k]
            col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
