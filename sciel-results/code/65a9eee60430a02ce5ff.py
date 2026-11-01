"""Matrix multiplication 2x3 * 3x8: rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 8
    deadline = time.time() + 110  # safety margin under 120s

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        A = np.random.randn(2, 3)
        B = np.random.randn(3, 8)
        C_expected = A @ B
        C_computed = np.zeros((2, 8))
        for c in range(U_np.shape[1]):
            U_mat = U_np[:, c].reshape(2, 3)
            V_mat = V_np[:, c].reshape(3, 8)
            W_vec = W_np[:, c]
            prod = U_mat @ V_mat
            for i in range(2):
                for k in range(8):
                    C_computed[i, k] += W_vec[k * 2 + i] * prod[i, k]
        return np.allclose(C_computed, C_expected, atol=1e-9)

    # Deterministic rank-36 construction:
    # Split B into two 3x4 halves (cols 0-3, 4-7).
    # For each half: 2x3*3x4 = (2x2*2x4 via 2 Strassen 2x2x2 blocks) + (2x1*1x4 outer products).
    # 2x2*2x4: two Strassen (rank 7 each) = 14.
    # 2x1*1x4: 4 outer-product terms (rank 4).
    # Per half: 18. Total: 36.
    rank = 36
    U = [[0.0] * rank for _ in range(6)]
    V = [[0.0] * rank for _ in range(24)]
    W = [[0.0] * rank for _ in range(16)]

    def add_strassen_2x2x2(base, a_r0, a_r1, b_c0, b_c1, b_k0):
        """Strassen 2x2*2x2: A rows a_r0,a_r1; A cols b_c0,b_c1; B rows b_c0,b_c1; B cols b_k0,b_k0+1.
        C[i,k] where i in {a_r0,a_r1}, k in {b_k0,b_k0+1}.
        M1 = (A00+A11)*(B00+B11)
        M2 = (A10+A11)*B00
        M3 = A00*(B10-B00)
        M4 = A11*(B01+B11)
        M5 = (A00+A01)*B11
        M6 = (A10-A00)*(B01-B11)
        M7 = (A01-A11)*(B00+B10)
        C00 = M1 - M3 + M4 - M7
        C01 = M4 + M5
        C10 = M2 + M3
        C11 = M1 - M2 + M6 - M7
        """
        # M1 = (A[a_r0,b_c0]+A[a_r1,b_c1])*(B[b_c0,b_k0]+B[b_c1,b_k0+1])
        U[a_r0 * 3 + b_c0][base] += 1; V[b_c0 * 8 + b_k0][base] += 1
        U[a_r1 * 3 + b_c1][base] += 1; V[b_c1 * 8 + b_k0 + 1][base] += 1
        W[b_k0 * 2 + a_r0][base] += 1       # C00 += M1
        W[(b_k0 + 1) * 2 + a_r1][base] += 1 # C11 += M1
        # M2 = (A[a_r1,b_c0]+A[a_r1,b_c1])*B[b_c0,b_k0]
        U[a_r1 * 3 + b_c0][base + 1] += 1; V[b_c0 * 8 + b_k0][base + 1] += 1
        U[a_r1 * 3 + b_c1][base + 1] += 1
        W[b_k0 * 2 + a_r1][base + 1] += 1   # C10 += M2
        # M3 = A[a_r0,b_c0]*(B[b_c1,b_k0]-B[b_c0,b_k0])
        U[a_r0 * 3 + b_c0][base + 2] += 1; V[b_c1 * 8 + b_k0][base + 2] += 1
        V[b_c0 * 8 + b_k0][base + 2] += -1
        W[b_k0 * 2 + a_r0][base + 2] += -1  # C00 -= M3
        W[b_k0 * 2 + a_r1][base + 2] += 1   # C10 += M3
        # M4 = A[a_r1,b_c1]*(B[b_c0,b_k0+1]+B[b_c1,b_k0+1])
        U[a_r1 * 3 + b_c1][base + 3] += 1; V[b_c0 * 8 + b_k0 + 1][base + 3] += 1
        V[b_c1 * 8 + b_k0 + 1][base + 3] += 1
        W[b_k0 * 2 + a_r0][base + 3] += 1   # C00 += M4
        W[(b_k0 + 1) * 2 + a_r0][base + 3] += 1  # C01 += M4
        # M5 = (A[a_r0,b_c0]+A[a_r0,b_c1])*B[b_c1,b_k0+1]
        U[a_r0 * 3 + b_c0][base + 4] += 1; V[b_c1 * 8 + b_k0 + 1][base + 4] += 1
        U[a_r0 * 3 + b_c1][base + 4] += 1
        W[(b_k0 + 1) * 2 + a_r0][base + 4] += 1  # C01 += M5
        # M6 = (A[a_r1,b_c0]-A[a_r0,b_c0])*(B[b_c0,b_k0+1]-B[b_c1,b_k0+1])
        U[a_r1 * 3 + b_c0][base + 5] += 1; V[b_c0 * 8 + b_k0 + 1][base + 5] += 1
        U[a_r0 * 3 + b_c0][base + 5] += -1
        V[b_c1 * 8 + b_k0 + 1][base + 5] += -1
        W[(b_k0 + 1) * 2 + a_r1][base + 5] += 1  # C11 += M6
        # M7 = (A[a_r0,b_c1]-A[a_r1,b_c1])*(B[b_c0,b_k0]+B[b_c1,b_k0])
        U[a_r0 * 3 + b_c1][base + 6] += 1; V[b_c0 * 8 + b_k0][base + 6] += 1
        U[a_r1 * 3 + b_c1][base + 6] += -1
        V[b_c1 * 8 + b_k0][base + 6] += 1
        W[b_k0 * 2 + a_r0][base + 6] += -1  # C00 -= M7
        W[(b_k0 + 1) * 2 + a_r1][base + 6] += -1  # C11 -= M7

    for half in range(2):
        bk = half * 4
        base = half * 18
        # 2x2*2x4: A rows 0,1; A cols 0,1; B rows 0,1; B cols bk..bk+3
        add_strassen_2x2x2(base, 0, 1, 0, 1, bk)
        add_strassen_2x2x2(base + 7, 0, 1, 0, 1, bk + 2)
        # 2x1*1x4: A col 2; B row 2; B cols bk..bk+3
        for dk in range(4):
            c = base + 14 + dk
            U[0 * 3 + 2][c] += 1
            U[1 * 3 + 2][c] += 1
            V[2 * 8 + bk + dk][c] += 1
            W[(bk + dk) * 2 + 0][c] += 1
            W[(bk + dk) * 2 + 1][c] += 1

    if time.time() > deadline:
        return schoolbook()

    if verify(U, V, W):
        return {"U": U, "V": V, "W": W}

    return schoolbook()
# EVOLVE-BLOCK-END
