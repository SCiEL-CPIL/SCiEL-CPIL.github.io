"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x2 * 2x11.
    # Split the 11 columns into five pairs and one single.
    # Each pair uses Strassen (rank 7), the single uses schoolbook (rank 4).
    # Total rank = 5*7 + 4 = 39.
    import time
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # We'll build the decomposition by defining the tensor coefficients.
    # The multiplication tensor has entries T[i, j, k] where:
    #   i in [0, n) = row of A and row of C
    #   j in [0, m) = col of A and row of B
    #   k in [0, p) = col of B and col of C
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) of C.
    # Actually, for matrix multiplication C = A*B:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T has T[i,j,k] = 1 for all valid i,j,k.
    # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Note the W convention: W[k*n + i, c]

    # Strassen for 2x2 * 2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11) B00
    # M3 = A00 (B10-B11)
    # M4 = A11 (B01+B11)
    # M5 = (A00+A01) B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6

    # For a 2-column block (columns k1, k2 of B, hence k1, k2 of C):
    # B is 2x2: B[j,k1], B[j,k2] for j in {0,1}
    # We apply Strassen to the 2x2*2x2 problem where:
    #   "B00" = B[0,k1], "B01" = B[0,k2]
    #   "B10" = B[1,k1], "B11" = B[1,k2]
    #
    # U factors (for A, indexed by i*m+j):
    #   M1: A00+A11 -> U[0]=1, U[3]=1
    #   M2: A10+A11 -> U[2]=1, U[3]=1
    #   M3: A00     -> U[0]=1
    #   M4: A11     -> U[3]=1
    #   M5: A00+A01 -> U[0]=1, U[1]=1
    #   M6: A10-A00 -> U[2]=1, U[0]=-1
    #   M7: A01-A11 -> U[1]=1, U[3]=-1
    #
    # V factors (for B, indexed by j*p+k; for this block, only k1,k2 are nonzero):
    #   M1: B00+B11 -> V[0*p+k1]=1, V[1*p+k2]=1
    #   M2: B00     -> V[0*p+k1]=1
    #   M3: B10-B11 -> V[1*p+k1]=1, V[1*p+k2]=-1
    #   M4: B01+B11 -> V[0*p+k2]=1, V[1*p+k2]=1
    #   M5: B11     -> V[1*p+k2]=1
    #   M6: B00+B10 -> V[0*p+k1]=1, V[1*p+k1]=1
    #   M7: B01+B11 -> V[0*p+k2]=1, V[1*p+k2]=1
    #
    # W factors (for C, indexed by k*n+i):
    #   C00 = M1+M4-M5+M7: W[k1*n+0] += 1 (M1), W[k1*n+0] += 1 (M4), W[k1*n+0] += -1 (M5), W[k1*n+0] += 1 (M7)
    #   Wait, W is accumulated over all M's. For each M_c, W has specific entries.
    #   M1: C00 and C11 -> W[k1*n+0]=1, W[k2*n+1]=1
    #   M2: C10 -> W[k1*n+1]=1
    #   M3: C01 and C11 -> W[k2*n+0]=1, W[k2*n+1]=1
    #   M4: C00 and C10 -> W[k1*n+0]=1, W[k1*n+1]=1
    #   M5: C01 and C00 -> W[k2*n+0]=1, W[k1*n+0]=-1
    #   M6: C11 -> W[k2*n+1]=1
    #   M7: C00 -> W[k1*n+0]=1

    rank = 5 * 7 + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def add_strassen_block(k1, k2, col_offset):
        """Add a Strassen block for columns k1, k2 starting at col_offset."""
        # M1
        U[0][col_offset + 0] += 1
        U[3][col_offset + 0] += 1
        V[0 * p + k1][col_offset + 0] += 1
        V[1 * p + k2][col_offset + 0] += 1
        W[k1 * n + 0][col_offset + 0] += 1
        W[k2 * n + 1][col_offset + 0] += 1
        # M2
        U[2][col_offset + 1] += 1
        U[3][col_offset + 1] += 1
        V[0 * p + k1][col_offset + 1] += 1
        W[k1 * n + 1][col_offset + 1] += 1
        # M3
        U[0][col_offset + 2] += 1
        V[1 * p + k1][col_offset + 2] += 1
        V[1 * p + k2][col_offset + 2] += -1
        W[k2 * n + 0][col_offset + 2] += 1
        W[k2 * n + 1][col_offset + 2] += 1
        # M4
        U[3][col_offset + 3] += 1
        V[0 * p + k2][col_offset + 3] += 1
        V[1 * p + k2][col_offset + 3] += 1
        W[k1 * n + 0][col_offset + 3] += 1
        W[k1 * n + 1][col_offset + 3] += 1
        # M5
        U[0][col_offset + 4] += 1
        U[1][col_offset + 4] += 1
        V[1 * p + k2][col_offset + 4] += 1
        W[k2 * n + 0][col_offset + 4] += 1
        W[k1 * n + 0][col_offset + 4] += -1
        # M6
        U[2][col_offset + 5] += 1
        U[0][col_offset + 5] += -1
        V[0 * p + k1][col_offset + 5] += 1
        V[1 * p + k1][col_offset + 5] += 1
        W[k2 * n + 1][col_offset + 5] += 1
        # M7
        U[1][col_offset + 6] += 1
        U[3][col_offset + 6] += -1
        V[0 * p + k2][col_offset + 6] += 1
        V[1 * p + k2][col_offset + 6] += 1
        W[k1 * n + 0][col_offset + 6] += 1

    def add_schoolbook_column(k, col_offset):
        """Add schoolbook for a single column k (4 terms)."""
        # A00*B0k -> C0k
        U[0][col_offset + 0] += 1
        V[0 * p + k][col_offset + 0] += 1
        W[k * n + 0][col_offset + 0] += 1
        # A01*B1k -> C0k
        U[1][col_offset + 1] += 1
        V[1 * p + k][col_offset + 1] += 1
        W[k * n + 0][col_offset + 1] += 1
        # A10*B0k -> C1k
        U[2][col_offset + 2] += 1
        V[0 * p + k][col_offset + 2] += 1
        W[k * n + 1][col_offset + 2] += 1
        # A11*B1k -> C1k
        U[3][col_offset + 3] += 1
        V[1 * p + k][col_offset + 3] += 1
        W[k * n + 1][col_offset + 3] += 1

    col = 0
    # Five 2-column blocks: (0,1), (2,3), (4,5), (6,7), (8,9)
    for g in range(5):
        k1 = 2 * g
        k2 = 2 * g + 1
        add_strassen_block(k1, k2, col)
        col += 7

    # One single column: 10
    add_schoolbook_column(10, col)
    col += 4

    # Verify: check that the decomposition is exact
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For all i,j,k, T should be 1.
    # We'll do a quick numerical check.
    import numpy as np
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-10:
        # Fallback to schoolbook if verification fails
        rank_fb = n * m * p
        U_fb = [[0] * rank_fb for _ in range(n * m)]
        V_fb = [[0] * rank_fb for _ in range(m * p)]
        W_fb = [[0] * rank_fb for _ in range(p * n)]
        col_fb = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][col_fb] = 1
                    V_fb[j * p + k][col_fb] = 1
                    W_fb[k * n + i][col_fb] = 1
                    col_fb += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
