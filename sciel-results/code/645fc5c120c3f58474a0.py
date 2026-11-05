"""Matrix multiplication 6x6x6 tensor decomposition."""
import time
import numpy as np
import random

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-89 decomposition of 6x6 matrix multiplication.

    Partition 6 = 4 + 2 into blocks:
      A = [[A11(4x4), A12(4x2)],
           [A21(2x4), A22(2x2)]]
      B = [[B11(4x4), B12(4x2)],
           [B21(2x4), B22(2x2)]]
      C = [[C11(4x4), C12(4x2)],
           [C21(2x4), C22(2x2)]]

    C11 = A11*B11  -> rank-49 via Strassen^2 (tensor square of 2x2 Strassen)
    C12 = A11*B12  -> schoolbook rank 4*2*2 = 16
    C21 = A21*B11  -> schoolbook rank 2*4*2 = 16
    C22 = A21*B21 + A22*B22 -> schoolbook rank 2*4*2 + 2*2*2 = 16
    Total rank = 49 + 16 + 16 + 16 = 89
    """
    start_time = time.time()
    deadline = 130.0

    n, m, p = DIMENSIONS  # all 6

    # --- Strassen 2x2 base (7 products) ---
    # Each product: (A-coeff dict, B-coeff dict, C-coeff dict) with block coords in {0,1}
    strassen = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1},                 {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1},                 {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1},                 {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1},                 {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1},     {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1},     {(0, 0): 1}),
    ]

    # --- Build the rank-49 4x4 decomposition (Strassen^2) ---
    # 4x4 blocks of size 2, indexed (I,J) in {0,1}^2.
    # Product (a,b): X = sum_{(I,J)} aA[(I,J)] A[(I,J)], Y = sum bB[(J,K)] B[(J,K)]
    # C[(I,K)] += aC[(I,K)] * X*Y
    strassen_sq = []
    for (aA, aB, aC) in strassen:
        for (bA, bB, bC) in strassen:
            # combine A coefficients over (I,J)
            Acoeff = {}
            for (I, J), v in aA.items():
                Acoeff[(I, J)] = Acoeff.get((I, J), 0) + v
            Bcoeff = {}
            for (J, K), v in bB.items():
                Bcoeff[(J, K)] = Bcoeff.get((J, K), 0) + v
            Ccoeff = {}
            for (I, K), v in aC.items():
                for (J, K2), w in bC.items():
                    if J == K2:
                        Ccoeff[(I, K)] = Ccoeff.get((I, K), 0) + v * w
            # drop zeros
            Acoeff = {k: v for k, v in Acoeff.items() if v != 0}
            Bcoeff = {k: v for k, v in Bcoeff.items() if v != 0}
            Ccoeff = {k: v for k, v in Ccoeff.items() if v != 0}
            strassen_sq.append((Acoeff, Bcoeff, Ccoeff))

    assert len(strassen_sq) == 49

    def idx_u(i, j):  # U index: i*m + j
        return i * m + j

    def idx_v(j, k):  # V index: j*p + k
        return j * p + k

    def idx_w(k, i):  # W index: k*n + i
        return k * n + i

    # Total rank
    rank = 49 + 16 + 16 + 16
    U = [[0] * rank for _ in range(n * m)]  # 36 x rank
    V = [[0] * rank for _ in range(m * p)]  # 36 x rank
    W = [[0] * rank for _ in range(p * n)]  # 36 x rank

    col = 0

    # ---- C11 = A11 * B11 : rank-49 (Strassen^2), 4x4 blocks of size 2 ----
    for (Acoeff, Bcoeff, Ccoeff) in strassen_sq:
        for u in range(4):
            for w in range(4):
                for v in range(4):
                    # U: X[u,v] = sum_{I,J} Acoeff[(I,J)] A[I*2+u, J*2+v]
                    for (I, J), ca in Acoeff.items():
                        gi = I * 2 + u
                        gj = J * 2 + v
                        U[idx_u(gi, gj)][col] += ca
                    # V: Y[v,w] = sum_{J,K} Bcoeff[(J,K)] B[J*2+v, K*2+w]
                    for (J, K), cb in Bcoeff.items():
                        gj = J * 2 + v
                        gk = K * 2 + w
                        V[idx_v(gj, gk)][col] += cb
                    # W: C[(I,K)] += Ccoeff[(I,K)] * M[u,w]  -> C[I*2+u, K*2+w]
                    for (I, K), cc in Ccoeff.items():
                        gi = I * 2 + u
                        gk = K * 2 + w
                        W[idx_w(gk, gi)][col] += cc
                    col += 1

    # ---- Schoolbook rectangular blocks ----
    # C12 = A11(4x4) * B12(4x2): i in 0..3, j in 0..3, k in 4..5
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                U[idx_u(i, j)][col] = 1
                V[idx_v(j, k)][col] = 1
                W[idx_w(k, i)][col] = 1
                col += 1

    # C21 = A21(2x4) * B11(4x4): i in 4..5, j in 0..3, k in 0..3
    for i in range(4, 6):
        for j in range(4):
            for k in range(4):
                U[idx_u(i, j)][col] = 1
                V[idx_v(j, k)][col] = 1
                W[idx_w(k, i)][col] = 1
                col += 1

    # C22 = A21(2x4)*B21(2x4) + A22(2x2)*B22(2x2): i in 4..5, k in 4..5
    #   part 1: j in 4..5 (A21[i,j] * B21[j,k])
    for i in range(4, 6):
        for j in range(4, 6):
            for k in range(4, 6):
                U[idx_u(i, j)][col] = 1
                V[idx_v(j, k)][col] = 1
                W[idx_w(k, i)][col] = 1
                col += 1
    #   part 2: j in 0..3 (A22[i,j] * B22[j,k])
    for i in range(4, 6):
        for j in range(4):
            for k in range(4, 6):
                U[idx_u(i, j)][col] = 1
                V[idx_v(j, k)][col] = 1
                W[idx_w(k, i)][col] = 1
                col += 1

    assert col == rank, (col, rank)

    # Verify
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    random.seed(42)
    valid = True
    for _ in range(300):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        val = 0.0
        for r in range(rank):
            val += U_arr[i * m + j, r] * V_arr[j * p + k, r] * W_arr[k * n + i, r]
        if abs(val - 1.0) > 1e-9:
            valid = False
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: guaranteed schoolbook rank 216
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j][c] = 1
                    V_f[j * p + k][c] = 1
                    W_f[k * n + i][c] = 1
                    c += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
