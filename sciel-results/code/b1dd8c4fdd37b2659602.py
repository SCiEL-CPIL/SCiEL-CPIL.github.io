"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 14
    # We decompose the 2x2 * 2x14 multiplication into 7 independent 2x2 * 2x2 multiplications.
    # For each block c (0..6), we use Strassen's rank-7 decomposition.
    # Total rank = 7 * 7 = 49.
    rank = 49
    U = [[0.0] * rank for _ in range(n * m)]  # 4 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 28 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 28 rows

    # Strassen coefficients:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Output:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For each block c and each Strassen term s (0..6):
    for c in range(7):
        for s in range(7):
            col = c * 7 + s
            # U contributions (A side):
            if s == 0:  # M1
                U[0][col] += 1  # A11
                U[3][col] += 1  # A22
            elif s == 1:  # M2
                U[2][col] += 1  # A21
                U[3][col] += 1  # A22
            elif s == 2:  # M3
                U[0][col] += 1  # A11
            elif s == 3:  # M4
                U[3][col] += 1  # A22
            elif s == 4:  # M5
                U[0][col] += 1  # A11
                U[1][col] += 1  # A12
            elif s == 5:  # M6
                U[2][col] += 1  # A21
                U[0][col] -= 1  # -A11
            elif s == 6:  # M7
                U[1][col] += 1  # A12
                U[3][col] -= 1  # -A22

            # V contributions (B side):
            # B11 is V[0*c*2 + 0] = V[0]
            # B12 is V[1*c*2 + 0] = V[14]... wait, let me re-check indexing.
            # V has shape (m*p, rank) = (28, rank)
            # V[j*p + k, col] where j is row of B (0 or 1), k is col of B (0..13)
            # For block c, the B block is 2x2 with columns 2c and 2c+1.
            # B11 of block c: j=0, k=2c -> V[0*14 + 2c] = V[2c]
            # B12 of block c: j=0, k=2c+1 -> V[2c+1]
            # B21 of block c: j=1, k=2c -> V[14 + 2c]
            # B22 of block c: j=1, k=2c+1 -> V[14 + 2c + 1]
            
            b11 = 2 * c
            b12 = 2 * c + 1
            b21 = 14 + 2 * c
            b22 = 14 + 2 * c + 1

            if s == 0:  # M1: B11+B22
                V[b11][col] += 1
                V[b22][col] += 1
            elif s == 1:  # M2: B11
                V[b11][col] += 1
            elif s == 2:  # M3: B12-B22
                V[b12][col] += 1
                V[b22][col] -= 1
            elif s == 3:  # M4: B21-B11
                V[b21][col] += 1
                V[b11][col] -= 1
            elif s == 4:  # M5: B22
                V[b22][col] += 1
            elif s == 5:  # M6: B11+B12
                V[b11][col] += 1
                V[b12][col] += 1
            elif s == 6:  # M7: B21+B22
                V[b21][col] += 1
                V[b22][col] += 1

            # W contributions (C side):
            # C has shape (p*n, rank) = (28, rank)
            # W[k*n + i, col] where i is row of C (0 or 1), k is col of C (0..13)
            # For block c, the C block is 2x2 with columns 2c and 2c+1.
            # C11 of block c: i=0, k=2c -> W[2c*2 + 0] = W[4c]
            # C12 of block c: i=0, k=2c+1 -> W[4c+1]
            # C21 of block c: i=1, k=2c -> W[4c+2]
            # C22 of block c: i=1, k=2c+1 -> W[4c+3]
            
            c11 = 4 * c
            c12 = 4 * c + 2
            c21 = 4 * c + 1
            c22 = 4 * c + 3

            if s == 0:  # M1: C11 +1, C22 +1
                W[c11][col] += 1
                W[c22][col] += 1
            elif s == 1:  # M2: C21 +1, C22 -1
                W[c21][col] += 1
                W[c22][col] -= 1
            elif s == 2:  # M3: C12 +1, C22 +1
                W[c12][col] += 1
                W[c22][col] += 1
            elif s == 3:  # M4: C11 +1, C21 +1
                W[c11][col] += 1
                W[c21][col] += 1
            elif s == 4:  # M5: C11 -1, C12 +1
                W[c11][col] -= 1
                W[c12][col] += 1
            elif s == 5:  # M6: C22 +1
                W[c22][col] += 1
            elif s == 6:  # M7: C11 +1
                W[c11][col] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
