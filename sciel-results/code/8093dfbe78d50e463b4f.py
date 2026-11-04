"""
Rank-189 decomposition of 6x6x6 matrix multiplication via Kronecker product
of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7).

Construction:
- Partition 6x6 matrices into 3x3 blocks of 2x2 submatrices
- Use schoolbook for the 3x3 block structure (27 terms)
- Use Strassen within each 2x2 block product (7 terms)
- Total rank: 27 * 7 = 189

The W matrix follows the official trace convention W[k*n+i, r].
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using Kronecker product of 3x3 schoolbook (rank 27) and 2x2 Strassen (rank 7).
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin

    n, m, p = DIMENSIONS  # 6, 6, 6

    # Define 2x2 Strassen decomposition
    # U22: shape (4, 7) where row = i*2+j for A[i,j], col = rank term
    # V22: shape (4, 7) where row = j*2+k for B[j,k], col = rank term
    # W22: shape (4, 7) where row = k*2+i for C[i,k], col = rank term

    U22 = np.zeros((4, 7), dtype=float)
    V22 = np.zeros((4, 7), dtype=float)
    W22 = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[1, 1] = 1; W22[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[2, 2] = 1; W22[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1; W22[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1

    # Define 3x3 schoolbook decomposition
    U33 = np.zeros((9, 27), dtype=float)
    V33 = np.zeros((9, 27), dtype=float)
    W33 = np.zeros((9, 27), dtype=float)

    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U33[i * 3 + j, col] = 1
                V33[j * 3 + k, col] = 1
                W33[k * 3 + i, col] = 1
                col += 1

    # Form the Kronecker product decomposition for 6x6
    rank = 27 * 7  # 189

    U66 = np.zeros((36, rank), dtype=float)
    V66 = np.zeros((36, rank), dtype=float)
    W66 = np.zeros((36, rank), dtype=float)

    # Precompute nonzero entries for efficiency
    # For each column in U33, V33, W33, store (row, value) pairs
    U33_nz = {}
    V33_nz = {}
    W33_nz = {}
    for c in range(27):
        U33_nz[c] = [(r, U33[r, c]) for r in range(9) if U33[r, c] != 0]
        V33_nz[c] = [(r, V33[r, c]) for r in range(9) if V33[r, c] != 0]
        W33_nz[c] = [(r, W33[r, c]) for r in range(9) if W33[r, c] != 0]

    U22_nz = {}
    V22_nz = {}
    W22_nz = {}
    for c in range(7):
        U22_nz[c] = [(r, U22[r, c]) for r in range(4) if U22[r, c] != 0]
        V22_nz[c] = [(r, V22[r, c]) for r in range(4) if V22[r, c] != 0]
        W22_nz[c] = [(r, W22[r, c]) for r in range(4) if W22[r, c] != 0]

    # Build final matrices using Kronecker product structure
    for c33 in range(27):
        for c22 in range(7):
            c_final = c33 * 7 + c22

            # U: target_row = (i3*2+i2)*6 + (j3*2+j2)
            # U33 row = i3*3+j3, U22 row = i2*2+j2
            for (r3, v3) in U33_nz[c33]:
                i3, j3 = divmod(r3, 3)
                for (r2, v2) in U22_nz[c22]:
                    i2, j2 = divmod(r2, 2)
                    target_row = (i3 * 2 + i2) * 6 + (j3 * 2 + j2)
                    U66[target_row, c_final] = v3 * v2

            # V: target_row = (j3*2+j2)*6 + (k3*2+k2)
            # V33 row = j3*3+k3, V22 row = j2*2+k2
            for (r3, v3) in V33_nz[c33]:
                j3, k3 = divmod(r3, 3)
                for (r2, v2) in V22_nz[c22]:
                    j2, k2 = divmod(r2, 2)
                    target_row = (j3 * 2 + j2) * 6 + (k3 * 2 + k2)
                    V66[target_row, c_final] = v3 * v2

            # W: target_row = (k3*2+k2)*6 + (i3*2+i2)
            # W33 row = k3*3+i3, W22 row = k2*2+i2
            for (r3, v3) in W33_nz[c33]:
                k3, i3 = divmod(r3, 3)
                for (r2, v2) in W22_nz[c22]:
                    k2, i2 = divmod(r2, 2)
                    target_row = (k3 * 2 + k2) * 6 + (i3 * 2 + i2)
                    W66[target_row, c_final] = v3 * v2

    # Verify correctness with vectorized approach
    if time.time() < deadline - 3:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m))
        B_test = np.random.randint(-5, 6, size=(m, p))
        C_expected = A_test @ B_test

        C_computed = np.zeros((n, p), dtype=float)
        for r in range(rank):
            u_col = U66[:, r]
            v_col = V66[:, r]
            w_col = W66[:, r]
            Au = (A_test * u_col.reshape(n, m))
            Bv = (B_test * v_col.reshape(m, p))
            w_mat = w_col.reshape(p, n).T
            C_computed += w_mat * (Au @ Bv)

        if not np.allclose(C_computed, C_expected, atol=1e-6):
            # Fallback to schoolbook rank 216
            rank_school = n * m * p
            U_fb = np.zeros((n * m, rank_school), dtype=float)
            V_fb = np.zeros((m * p, rank_school), dtype=float)
            W_fb = np.zeros((p * n, rank_school), dtype=float)
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_fb[i * m + j, column] = 1
                        V_fb[j * p + k, column] = 1
                        W_fb[k * n + i, column] = 1
                        column += 1
            return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U66.tolist(), "V": V66.tolist(), "W": W66.tolist()}
# EVOLVE-BLOCK-END
