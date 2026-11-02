"""Strassen-based rank reduction for 2x2x15 matrix multiplication."""
import time
import sys

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 2, 15

    # Strassen decomposition for 2x2x2: rank 7
    # The 7 products are:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # U indices: i*m+j -> i*2+j for A[i][j]
    # V indices: j*p+k -> j*2+k for B[j][k]
    # W indices: k*n+i -> k*2+i for C[k][i] (note: C is 2x2, so k in {0,1}, i in {0,1})
    # W[k*n+i] = W[k*2+i]

    # Strassen U, V, W for 2x2x2 (each is 4x7 or 4x7 or 4x7)
    # U: 4 rows (A elements), 7 columns (products)
    # V: 4 rows (B elements), 7 columns
    # W: 4 rows (C elements), 7 columns

    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 1, 0, 0, 0, 0, -1],   # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, 1],    # A11
    ]
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 0],    # B00
        [0, 0, 1, -1, 1, 1, 1],   # B01
        [0, 0, -1, 1, 0, 0, 0],   # B10
        [1, 0, -1, 0, 1, 0, 1],   # B11
    ]
    strassen_W = [
        [1, 0, 0, 0, -1, 0, -1],  # C00 (k=0,i=0)
        [0, 0, 1, 0, 1, 0, 0],    # C01 (k=0,i=1)
        [0, 1, 0, 1, 0, 0, 0],    # C10 (k=1,i=0)
        [1, -1, 1, 0, 0, 0, 1],   # C11 (k=1,i=1)
    ]

    # Now decompose p=15 into 7 blocks of size 2 and 1 block of size 1
    # For each block of size 2, we use Strassen (rank 7)
    # For the block of size 1, we use schoolbook (rank 4)
    #
    # Total rank = 7*7 + 4 = 49 + 4 = 53

    # But wait - we need to be more careful. The matrix B is 2x15.
    # We split the 15 columns into groups. Each group of 2 columns uses Strassen,
    # and the remaining 1 column uses schoolbook.
    #
    # Actually, let me reconsider. The standard approach:
    # A is 2x2, B is 2x15, C is 2x15.
    # We can think of B as having 15 column pairs... no, 15 = 7*2 + 1.
    #
    # For each pair of columns (2j, 2j+1) of B, we multiply A (2x2) by B[:, 2j:2j+2] (2x2)
    # to get C[:, 2j:2j+2] (2x2). This uses Strassen with rank 7.
    # For the last single column (column 14), we multiply A (2x2) by B[:, 14:15] (2x1)
    # to get C[:, 14:15] (2x1). This uses schoolbook with rank 4 (since it's 2x2x1).
    #
    # Wait, for 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    # C[0] = A[0][0]*B[0] + A[0][1]*B[1]
    # C[1] = A[1][0]*B[0] + A[1][1]*B[1]
    # This is rank 4 (each C element needs 2 products, 2 elements = 4 products).
    # Actually it's just the standard bilinear form, rank = 2*2*1 = 4.
    #
    # So total rank = 7 * 7 + 4 = 53.

    # Let's build the full U, V, W matrices.
    # U has n*m = 4 rows, rank columns
    # V has m*p = 30 rows, rank columns
    # W has p*n = 30 rows, rank columns

    rank = 7 * 7 + 4  # 53

    U = [[0] * rank for _ in range(n * m)]  # 4 x 53
    V = [[0] * rank for _ in range(m * p)]  # 30 x 53
    W = [[0] * rank for _ in range(p * n)]  # 30 x 53

    col = 0
    for j in range(7):
        # Block j: columns 2*j and 2*j+1 of B
        # Uses Strassen rank 7
        for s in range(7):
            # U contribution (same for all blocks since A is the same)
            for i in range(4):
                U[i][col] += strassen_U[i][s]
            # V contribution: B rows j=0,1 for columns 2*j and 2*j+1
            # V index: b_row * p + b_col
            # B[0][2*j] -> V[0*15 + 2*j] = V[2*j]
            # B[0][2*j+1] -> V[2*j+1]
            # B[1][2*j] -> V[1*15 + 2*j] = V[15 + 2*j]
            # B[1][2*j+1] -> V[15 + 2*j + 1]
            V[2 * j][col] += strassen_V[0][s]       # B[0][2*j]
            V[2 * j + 1][col] += strassen_V[1][s]   # B[0][2*j+1]
            V[15 + 2 * j][col] += strassen_V[2][s]  # B[1][2*j]
            V[15 + 2 * j + 1][col] += strassen_V[3][s]  # B[1][2*j+1]
            # W contribution: C columns 2*j and 2*j+1
            # W index: c_col * n + c_row = c_col * 2 + c_row
            # C[0][2*j] -> W[2*j*2 + 0] = W[4*j]
            # C[1][2*j] -> W[4*j + 1]
            # C[0][2*j+1] -> W[(2*j+1)*2 + 0] = W[4*j + 2]
            # C[1][2*j+1] -> W[4*j + 3]
            W[4 * j][col] += strassen_W[0][s]       # C[0][2*j]
            W[4 * j + 1][col] += strassen_W[1][s]   # C[1][2*j]
            W[4 * j + 2][col] += strassen_W[2][s]   # C[0][2*j+1]
            W[4 * j + 3][col] += strassen_W[3][s]   # C[1][2*j+1]
            col += 1

    # Last block: column 14 of B (single column)
    # Schoolbook for 2x2x1: rank 4
    # C[0][14] = A[0][0]*B[0][14] + A[0][1]*B[1][14]
    # C[1][14] = A[1][0]*B[0][14] + A[1][1]*B[1][14]
    #
    # Products:
    # P0 = A[0][0] * B[0][14]
    # P1 = A[0][1] * B[1][14]
    # P2 = A[1][0] * B[0][14]
    # P3 = A[1][1] * B[1][14]
    #
    # C[0][14] = P0 + P1
    # C[1][14] = P2 + P3

    # P0
    U[0][col] = 1  # A[0][0]
    V[14][col] = 1  # B[0][14]
    W[28][col] = 1  # C[0][14] -> W[14*2+0] = W[28]
    col += 1

    # P1
    U[1][col] = 1  # A[0][1]
    V[29][col] = 1  # B[1][14] -> V[15+14] = V[29]
    W[28][col] = 1  # C[0][14]
    col += 1

    # P2
    U[2][col] = 1  # A[1][0]
    V[14][col] = 1  # B[0][14]
    W[29][col] = 1  # C[1][14] -> W[14*2+1] = W[29]
    col += 1

    # P3
    U[3][col] = 1  # A[1][1]
    V[29][col] = 1  # B[1][14]
    W[29][col] = 1  # C[1][14]
    col += 1

    assert col == rank, f"Expected rank {rank}, got {col}"

    # Verify the decomposition is correct using random test vectors
    import numpy as np
    np.random.seed(42)
    A = np.random.randint(-3, 4, size=(n, m)).astype(float)
    B = np.random.randint(-3, 4, size=(m, p)).astype(float)
    C_expected = A @ B

    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    # The tensor decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta-like structure
    # Actually, the bilinear form is: C[k,i] = sum_{j} A[i,j] * B[j,k]
    # The decomposition gives: C[k,i] = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) * W[k*n+i,c]
    # Wait no. Let me re-read the convention.
    #
    # The standard tensor decomposition for matrix multiplication:
    # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
    # where T_{ijk} corresponds to the (i,j,k) entry of the matmul tensor.
    #
    # For our case: A is (n x m), B is (m x p), C is (p x n).
    # C[k,i] = sum_j A[i,j] * B[j,k]
    #
    # The tensor is T[i,j,k] such that C[k,i] = sum_{i,j} A[i,j] B[j,k] T[i,j,k]... 
    # No, that's not right either.
    #
    # Let me use the convention from the task:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    #
    # The decomposition should satisfy:
    # For all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 if this is a valid "identity"
    #
    # Actually, the way it works:
    # C[k,i] = sum_j A[i,j] * B[j,k]
    # Using the decomposition:
    # C[k,i] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
    #
    # Wait, that doesn't look right dimensionally. Let me think again.
    #
    # The standard formulation: we want to compute C = A * B.
    # The bilinear algorithm says:
    # C[k,i] = sum_r (sum_{i',j'} U[i',j',r] * A[i',j']) * (sum_{j',k'} V[j',k',r] * B[j',k']) * W[k,i,r]
    #
    # Hmm, but the indices are different. Let me just verify numerically.
    #
    # The tensor T[i,j,k] = 1 for all valid i,j,k (since C[k,i] = sum_j A[i,j]*B[j,k],
    # the coefficient of A[i,j]*B[j,k] in C[k,i] is 1).
    #
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k.
    #
    # Let's verify this.

    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should be 1
    error = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > 1e-10:
                    error += abs(s - 1.0)

    if error > 1e-8:
        # Fallback to schoolbook if Strassen verification fails
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
    else:
        # Verify with actual matrix multiplication
        C_computed = np.zeros((p, n))
        for c in range(rank):
            u_col = U_arr[:, c]
            v_col = V_arr[:, c]
            w_col = W_arr[:, c]
            # u_col has 4 elements (A indices)
            # v_col has 30 elements (B indices)
            # w_col has 30 elements (C indices)
            #
            # The contribution to C from rank c:
            # For each (i,j,k): u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i]
            # This contributes to C[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        C_computed[k, i] += u_col[i*m+j] * v_col[j*p+k] * w_col[k*n+i] * A[i,j] * B[j,k]
        
        if np.max(np.abs(C_computed - C_expected)) > 1e-8:
            # Fallback to schoolbook
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

    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Not enough time, just return what we have (should already be valid)
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
