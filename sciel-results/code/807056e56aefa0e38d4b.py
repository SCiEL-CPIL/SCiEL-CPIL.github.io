"""
Strassen-based decomposition for 2x3 x 3x8 matrix multiplication.

Strategy:
  - Split the 2x3 matrix A into A1 (cols 0-1) and A2 (col 2).
  - Split the 3x8 matrix B into B1 (rows 0-1) and B2 (row 2).
  - The product C = A*B = A1*B1 + A2*B2.
  - A1*B1 is 2x2 * 2x8: split B1 into four 2x2 blocks, apply Strassen to each.
    Each 2x2*2x2 uses 7 multiplications. 4 blocks * 7 = 28.
  - A2*B2 is 2x1 * 1x8: schoolbook, 16 multiplications.
  Total rank = 28 + 16 = 44.
"""

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # 6 x 44
    V = [[0] * rank for _ in range(m * p)]   # 24 x 44
    W = [[0] * rank for _ in range(p * n)]   # 16 x 44

    col = 0

    # --- Part 1: A1 (2x2) * B1 (2x8), split B1 into four 2x2 blocks ---
    # A1 = [[A00, A01], [A10, A11]] where A00=A[0,0], A01=A[0,1], A10=A[1,0], A11=A[1,1]
    # B1 has rows 0,1 and columns 0..7.
    # For each 2x2 block B[j1:j3, k1:k3] where j1=0,j2=1, k1,k2 vary:
    #   Strassen 7 products z1..z7
    #   Output C[i, k] += contribution

    def add_strassen_2x2(k1, k2):
        """Add 7 Strassen multiplications for A1 * B[0:2, k1:k2]."""
        nonlocal col
        # B block indices: B[0, k1], B[0, k2], B[1, k1], B[1, k2]
        # U indices: A00=U[0], A01=U[1], A10=U[3], A11=U[4]
        # V indices: B[0,k1]=V[k1], B[0,k2]=V[k2], B[1,k1]=V[8+k1], B[1,k2]=V[8+k2]
        # W indices: W[k*n + i] = W[2*i + k] for i in {0,1}, k in {k1,k2}

        # z1 = (A00+A11)*(B[0,k1]+B[1,k2])
        U[0][col] = 1; U[4][col] = 1
        V[k1][col] = 1; V[8+k2][col] = 1
        W[2*0+k1][col] += 1
        W[2*0+k2][col] += 1
        W[2*1+k1][col] += 1
        W[2*1+k2][col] += 1
        col += 1

        # z2 = (A10+A11)*B[0,k1]
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        W[2*1+k1][col] += 1
        W[2*1+k2][col] += 1
        col += 1

        # z3 = A00*(B[1,k1]-B[0,k1])
        U[0][col] = 1
        V[8+k1][col] = 1; V[k1][col] = -1
        W[2*0+k1][col] += 1
        W[2*0+k2][col] += 1
        col += 1

        # z4 = A11*(B[0,k2]-B[1,k1])
        U[4][col] = 1
        V[k2][col] = 1; V[8+k1][col] = -1
        W[2*1+k1][col] += 1
        W[2*1+k2][col] += 1
        col += 1

        # z5 = (A00+A01)*B[1,k2]
        U[0][col] = 1; U[1][col] = 1
        V[8+k2][col] = 1
        W[2*0+k2][col] += 1
        W[2*1+k2][col] += 1
        col += 1

        # z6 = (A10-A00)*(B[0,k1]+B[0,k2])
        U[3][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[k2][col] = 1
        W[2*1+k1][col] += 1
        W[2*1+k2][col] += 1
        col += 1

        # z7 = (A01-A11)*(B[1,k1]+B[1,k2])
        U[1][col] = 1; U[4][col] = -1
        V[8+k1][col] = 1; V[8+k2][col] = 1
        W[2*0+k1][col] += 1
        W[2*0+k2][col] += 1
        col += 1

    # Apply Strassen to four 2x2 blocks of B1
    add_strassen_2x2(0, 1)
    add_strassen_2x2(2, 3)
    add_strassen_2x2(4, 5)
    add_strassen_2x2(6, 7)

    # --- Part 2: A2 (2x1) * B2 (1x8), schoolbook ---
    # A2 = [A02, A12] where A02=A[0,2], A12=A[1,2]
    # B2 = B[2, 0:8]
    # C[i, k] += A[i,2] * B[2, k]
    # U indices: A02=U[2], A12=U[5]
    # V indices: B[2,k]=V[16+k]
    # W indices: W[2*i + k]
    for k in range(8):
        U[2][col] = 1
        V[16 + k][col] = 1
        W[2 * 0 + k][col] = 1
        col += 1
        U[5][col] = 1
        V[16 + k][col] = 1
        W[2 * 1 + k][col] = 1
        col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
