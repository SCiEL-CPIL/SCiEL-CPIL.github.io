import time
import numpy as np

DIMENSIONS = [2, 3, 7]


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 42

    # ---------- Trivial (schoolbook) decomposition as guaranteed fallback ----------
    U0 = [[0] * trivial_rank for _ in range(n * m)]
    V0 = [[0] * trivial_rank for _ in range(m * p)]
    W0 = [[0] * trivial_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i * m + j][col] = 1
                V0[j * p + k][col] = 1
                W0[k * n + i][col] = 1
                col += 1
    best_result = {"U": U0, "V": V0, "W": W0}
    best_rank = trivial_rank

    # ---------- Fast verification (numpy, vectorised) ----------
    def check(U_r, V_r, W_r):
        Ua = np.asarray(U_r, dtype=np.int64)
        Va = np.asarray(V_r, dtype=np.int64)
        Wa = np.asarray(W_r, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    prod = Ua[i * m + j] * Va[j * p + k] * Wa[k * n + i]
                    if int(np.sum(prod)) != 1:
                        return False
        return True

    # ---------- Strassen 2x2x2 decomposition (rank 7) ----------
    # For A (2x2), B (2x2), C = A*B.
    # U indexed by A[i*2+j], V by B[j*2+k], W by C[k*2+i] (trace convention W[k*n+i]).
    # Strassen factors M1..M7:
    #   M1=(A00+A11)(B01+B10)
    #   M2=(A10+A11)B00
    #   M3=A00(B10-B11)
    #   M4=A11(B01-B00)
    #   M5=(A00+A10)B11
    #   M6=(A10-A00)(B00+B01)
    #   M7=(A01-A11)(B10+B11)
    # C00=M1+M4-M5+M7 ; C01=M3+M5 ; C10=M2+M4 ; C11=M1-M2+M3+M6
    def strassen_block(U_b, V_b, W_b, base):
        # U_b is 4x7 (rows: A00,A01,A10,A11), V_b 4x7 (B00,B01,B10,B11), W_b 4x7 (C00,C01,C10,C11)
        a00, a01, a10, a11 = 0, 1, 2, 3
        b00, b01, b10, b11 = 0, 1, 2, 3
        c00, c01, c10, c11 = 0, 1, 2, 3
        # (U_row, V_row, W_row, sign) for each Strassen factor
        terms = [
            (a00, b01, c00, +1), (a11, b01, c00, +1),
            (a00, b10, c00, +1), (a11, b10, c00, +1),
            (a10, b00, c10, +1), (a11, b00, c10, +1),
            (a00, b10, c01, +1), (a00, b11, c01, -1),
            (a11, b01, c11, +1), (a11, b00, c11, -1),
            (a00, b11, c11, +1), (a10, b11, c11, +1),
            (a10, b00, c11, +1), (a10, b01, c11, +1),
            (a00, b00, c11, -1), (a00, b01, c11, -1),
            (a01, b10, c00, +1), (a11, b10, c00, -1),
            (a01, b11, c00, -1), (a11, b11, c00, +1),
        ]
        for (ur, vr, wr, s) in terms:
            U_b[ur][base] += s
            V_b[vr][base] += s
            W_b[wr][base] += s
        base += 1
        return base

    # ---------- Construct the hybrid decomposition ----------
    # Split A into 3 row-blocks by inner index j:
    #   Block A: j in {0,1}  -> 2x2 (rows i=0,1)  combined with B cols k in {0,1} -> 2x2x2 Strassen
    #   Block B: j in {0,1}  -> 2x2  combined with B col  k = 2        -> 2x1x2 schoolbook
    #   Block C: j = 2       -> 1x2 (rows i=0,1) combined with B cols k in {0,1,2} -> 1x2x3 schoolbook
    # For each (i,j,k) the single term contributes +1 to U[i*m+j]*V[j*p+k]*W[k*n+i].
    U = [[0] * trivial_rank for _ in range(n * m)]
    V = [[0] * trivial_rank for _ in range(m * p)]
    W = [[0] * trivial_rank for _ in range(p * n)]
    base = 0

    # --- Block A: i in {0,1}, j in {0,1}, k in {0,1} : 2x2x2 via Strassen ---
    # Local 2x2 A indices: (i,j) -> local a = i*2+j ; B (j,k) -> local b = j*2+k ; C (k,i) -> local c = k*2+i
    # We build local 4x7 matrices then map into global rows.
    U_loc = [[0] * 7 for _ in range(4)]
    V_loc = [[0] * 7 for _ in range(4)]
    W_loc = [[0] * 7 for _ in range(4)]
    base = strassen_block(U_loc, V_loc, W_loc, 0)  # base becomes 7
    # Map local rows to global
    for a in range(4):
        i = a // 2
        j = a % 2
        gU = i * m + j
        for t in range(7):
            U[gU][base + t] = U_loc[a][t]
    for b in range(4):
        j = b // 2
        k = b % 2
        gV = j * p + k
        for t in range(7):
            V[gV][base + t] = V_loc[b][t]
    for c in range(4):
        k = c // 2
        i = c % 2
        gW = k * n + i
        for t in range(7):
            W[gW][base + t] = W_loc[c][t]
    base += 7

    # --- Block B: i in {0,1}, j in {0,1}, k = 2 : 2x1x2 schoolbook (rank 4) ---
    for i in range(2):
        for j in range(2):
            U[i * m + j][base] = 1
            V[j * p + 2][base] = 1
            W[2 * n + i][base] = 1
            base += 1

    # --- Block C: i in {0,1}, j = 2, k in {0,1,2} : 1x2x3 schoolbook (rank 6) ---
    for i in range(2):
        for k in range(3):
            U[i * m + 2][base] = 1
            V[2 * p + k][base] = 1
            W[k * n + i][base] = 1
            base += 1

    # base is the achieved rank
    achieved_rank = base
    print("achieved_rank", achieved_rank)

    # Verify the hybrid decomposition
    if check(U, V, W):
        # Trim to achieved rank
        U = [row[:achieved_rank] for row in U]
        V = [row[:achieved_rank] for row in V]
        W = [row[:achieved_rank] for row in W]
        best_rank = achieved_rank
        best_result = {"U": U, "V": V, "W": W}
        print("hybrid valid, rank", best_rank)
    else:
        print("hybrid invalid, falling back to trivial rank 42")
        best_result = {"U": U0, "V": V0, "W": W0}
        best_rank = trivial_rank

    if time.time() > deadline - 5:
        return best_result

    return best_result
