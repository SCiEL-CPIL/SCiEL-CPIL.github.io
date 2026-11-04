"""
Design: 3x3 block partition of 6x6 matrices, each 2x2 block multiplied via Strassen.
This gives rank = 27 * 7 = 189.

The 6x6 matrices are partitioned into a 3x3 grid of 2x2 blocks:
- A[i,j] where i = 2*bi+ii, j = 2*bj+jj  for bi,bj in {0,1,2}, ii,jj in {0,1}
- B[j,k] where j = 2*bj+jj, k = 2*bk+kk
- C[i,k] where i = 2*bi+ii, k = 2*bk+kk, stored as W[k*6+i, r]

For each (bi,bj,bk) triple, we multiply the 2x2 block A[bi,bj] by B[bj,bk]
using Strassen's 7-multiplication algorithm, contributing to C[bi,bk].

Total rank: 27 (block positions) * 7 (Strassen) = 189.
"""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START

def _build_strassen_2x2():
    """Build the 2x2 Strassen decomposition (rank 7).
    
    Convention:
    - U: A[i,j] at row i*2+j
    - V: B[j,k] at row j*2+k
    - W: C[i,k] at row k*2+i (trace convention)
    """
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)

    # P1: (x11+x22)(y11+y22) -> c11+=1, c22+=1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # P2: (x21+x22)(y11) -> c21+=1, c22+=-1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # P3: (x11)(y12-y22) -> c12+=1, c22+=1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # P4: (x22)(y21-y11) -> c11+=1, c21+=1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # P5: (x11+x12)(y22) -> c11+=-1, c12+=1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # P6: (x21-x11)(y11+y12) -> c22+=1
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # P7: (x12-x22)(y21+y22) -> c11+=1
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W


def construct():
    """
    Construct a rank-189 decomposition of 6x6 * 6x6 -> 6x6 matrix multiplication
    using 3x3 block partition with Strassen 2x2 within each block.
    """
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit

    n, m, p = DIMENSIONS  # 6, 6, 6

    U22, V22, W22 = _build_strassen_2x2()

    rank = 27 * 7  # 189

    U66 = np.zeros((n * m, rank), dtype=float)  # (36, 189)
    V66 = np.zeros((m * p, rank), dtype=float)  # (36, 189)
    W66 = np.zeros((p * n, rank), dtype=float)  # (36, 189)

    # For each block position (bi, bj, bk) and Strassen term (r22)
    for bi in range(3):
        for bj in range(3):
            for bk in range(3):
                for r22 in range(7):
                    r_final = (bi * 9 + bj * 3 + bk) * 7 + r22

                    # U: A[i,j] where i=2*bi+ii, j=2*bj+jj
                    # U66 row = i*6+j = (2*bi+ii)*6 + (2*bj+jj)
                    # U22 row = ii*2+jj
                    for ii in range(2):
                        for jj in range(2):
                            u22_val = U22[ii * 2 + jj, r22]
                            if u22_val == 0:
                                continue
                            i = 2 * bi + ii
                            j = 2 * bj + jj
                            target_row = i * 6 + j
                            U66[target_row, r_final] = u22_val

                    # V: B[j,k] where j=2*bj+jj, k=2*bk+kk
                    # V66 row = j*6+k = (2*bj+jj)*6 + (2*bk+kk)
                    # V22 row = jj*2+kk
                    for jj in range(2):
                        for kk in range(2):
                            v22_val = V22[jj * 2 + kk, r22]
                            if v22_val == 0:
                                continue
                            j = 2 * bj + jj
                            k = 2 * bk + kk
                            target_row = j * 6 + k
                            V66[target_row, r_final] = v22_val

                    # W: C[i,k] -> W[k*6+i] where i=2*bi+ii, k=2*bk+kk
                    # W66 row = k*6+i = (2*bk+kk)*6 + (2*bi+ii)
                    # W22 row = kk*2+ii
                    for ii in range(2):
                        for kk in range(2):
                            w22_val = W22[kk * 2 + ii, r22]
                            if w22_val == 0:
                                continue
                            i = 2 * bi + ii
                            k = 2 * bk + kk
                            target_row = k * 6 + i
                            W66[target_row, r_final] = w22_val

    # Verify the decomposition
    if time.time() < deadline:
        np.random.seed(42)
        A_test = np.random.randint(-5, 6, size=(n, m)).astype(float)
        B_test = np.random.randint(-5, 6, size=(m, p)).astype(float)
        C_expected = A_test @ B_test

        # Vectorized verification
        # C[i,k] = sum_j sum_r A[i,j] * B[j,k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape for vectorized computation
        U_t = U66.T  # (rank, 36)
        V_t = V66.T  # (rank, 36)
        W_t = W66.T  # (rank, 36)

        A_flat = A_test.flatten()  # (36,)
        B_flat = B_test.flatten()  # (36,)

        # For each rank term r:
        #   contrib = sum_{i,j} A[i,j]*U[i*m+j, r] * sum_{j,k} B[j,k]*V[j*p+k, r] * sum_{i,k} W[k*n+i, r]*C[i,k]
        # This is equivalent to:
        #   u_r = A_flat @ U_t[r]  (scalar)
        #   v_r = B_flat @ V_t[r]  (scalar)
        #   Then C_computed += u_r * v_r * W_t[r] reshaped

        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_r = np.dot(A_flat, U_t[r])
            if u_r == 0:
                continue
            v_r = np.dot(B_flat, V_t[r])
            if v_r == 0:
                continue
            # W_t[r] has shape (36,), indexed by k*n+i
            W_r = W_t[r].reshape(p, n)  # W[k*n+i] -> reshape to (p, n)
            C_computed += u_r * v_r * W_r.T  # (n, p)

        if not np.allclose(C_computed, C_expected, atol=1e-9):
            # Fallback: schoolbook
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
