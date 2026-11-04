"""Combined program: rank-33 construction (schoolbook row-0 + 3x Strassen 2x2x2 blocks)."""
import numpy as np
import time

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    n, m, p = 2, 3, 6

    # Strategy:
    # Part 1: A[:, 0] (2x1) * B[0, :] (1x6) -> C[:, :]  (schoolbook, rank 12)
    # Part 2: A[:, 1:3] (2x2) * B[1:3, :] (2x6)
    #   Split B[1:3, :] into three 2x2 blocks by columns: (0,1), (2,3), (4,5)
    #   Each uses Strassen (rank 7) -> total 21
    # Grand total rank = 12 + 21 = 33

    rank = 33
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    # Part 1: A[i, 0] * B[0, k] -> C[i, k]
    for k in range(p):
        for i in range(n):
            U[i * m + 0, col] = 1.0
            V[0 * p + k, col] = 1.0
            W[k * n + i, col] = 1.0
            col += 1

    # Part 2: Three Strassen 2x2x2 blocks for A[:, 1:3] * B[1:3, block_cols]
    # A rows: i=0,1 ; A cols: j=1,2
    # B rows: j=1,2 ; B cols: (2t, 2t+1) for t=0,1,2
    # C rows: i=0,1 ; C cols: k=(2t, 2t+1)

    for t in range(3):
        # Column indices in B and C for this block
        k0 = 2 * t
        k1 = 2 * t + 1

        # Strassen formulas for C = A_sub * B_sub where:
        # A_sub[i,j] = A[i, 1+j] for i in {0,1}, j in {0,1}
        # B_sub[j,k] = B[1+j, k0+k] for j in {0,1}, k in {0,1}
        # C_sub[i,k] = C[i, k0+k]

        # M1 = (A00 + A11) * (B00 + B11)
        # U: A00 = U[0*m+1], A11 = U[1*m+2]
        # V: B00 = V[1*p+k0], B11 = V[2*p+k1]
        # W: C00 = W[k0*n+0], C11 = W[k1*n+1]
        U[0 * m + 1, col] += 1.0
        U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # M2 = (A10 + A11) * B00
        # U: A10 = U[1*m+1], A11 = U[1*m+2]
        # V: B00 = V[1*p+k0]
        # W: C10 = W[k0*n+1], C11 = W[k1*n+1]
        U[1 * m + 1, col] += 1.0
        U[1 * m + 2, col] += 1.0
        V[1 * p + k0, col] += 1.0
        W[k0 * n + 1, col] += 1.0
        W[k1 * n + 1, col] += -1.0
        col += 1

        # M3 = A00 * (B01 - B11)
        # U: A00 = U[0*m+1]
        # V: B01 = V[1*p+k1], B11 = V[2*p+k1]
        # W: C01 = W[k1*n+0], C11 = W[k1*n+1]
        U[0 * m + 1, col] += 1.0
        V[1 * p + k1, col] += 1.0
        V[2 * p + k1, col] += -1.0
        W[k1 * n + 0, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # M4 = A11 * (B10 - B00)
        # U: A11 = U[1*m+2]
        # V: B10 = V[2*p+k0], B00 = V[1*p+k0]
        # W: C00 = W[k0*n+0], C10 = W[k0*n+1]
        U[1 * m + 2, col] += 1.0
        V[2 * p + k0, col] += 1.0
        V[1 * p + k0, col] += -1.0
        W[k0 * n + 0, col] += 1.0
        W[k0 * n + 1, col] += 1.0
        col += 1

        # M5 = (A00 + A01) * B11
        # U: A00 = U[0*m+1], A01 = U[0*m+2]
        # V: B11 = V[2*p+k1]
        # W: C00 = W[k0*n+0], C01 = W[k1*n+0]
        U[0 * m + 1, col] += 1.0
        U[0 * m + 2, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += -1.0
        W[k1 * n + 0, col] += 1.0
        col += 1

        # M6 = (A10 - A00) * (B00 + B01)
        # U: A10 = U[1*m+1], A00 = U[0*m+1]
        # V: B00 = V[1*p+k0], B01 = V[1*p+k1]
        # W: C11 = W[k1*n+1]
        U[1 * m + 1, col] += 1.0
        U[0 * m + 1, col] += -1.0
        V[1 * p + k0, col] += 1.0
        V[1 * p + k1, col] += 1.0
        W[k1 * n + 1, col] += 1.0
        col += 1

        # M7 = (A01 - A11) * (B10 + B11)
        # U: A01 = U[0*m+2], A11 = U[1*m+2]
        # V: B10 = V[2*p+k0], B11 = V[2*p+k1]
        # W: C00 = W[k0*n+0]
        U[0 * m + 2, col] += 1.0
        U[1 * m + 2, col] += -1.0
        V[2 * p + k0, col] += 1.0
        V[2 * p + k1, col] += 1.0
        W[k0 * n + 0, col] += 1.0
        col += 1

    # Verify correctness: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    U_r = U.reshape(n, m, rank)
    V_r = V.reshape(m, p, rank)
    W_r = W.reshape(p, n, rank)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.sum(U_r[i, j, :] * V_r[j, k, :] * W_r[k, i, :])
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback: schoolbook rank 36
    if time.time() > deadline - 5.0:
        # Emergency: return whatever we have (shouldn't happen)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    rank_fb = n * m * p
    U_fb = np.zeros((n * m, rank_fb))
    V_fb = np.zeros((m * p, rank_fb))
    W_fb = np.zeros((p * n, rank_fb))
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j, c] = 1.0
                V_fb[j * p + k, c] = 1.0
                W_fb[k * n + i, c] = 1.0
                c += 1
    return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
# EVOLVE-BLOCK-END
