import time
import numpy as np

_DEADLINE = 115.0
_start_time = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def construct():
    n, m, p = 2, 3, 9
    rank = 50

    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    # Part 1: Strassen on 4 blocks of 2x2 (B columns 0-7, rows 0-1)
    # A_left = A[0:2, 0:2], B_left = B[0:2, 0:8]
    # A indices: a=A[0,0]=0, b=A[0,1]=1, c=A[1,0]=3, d=A[1,1]=4
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        v_e = k0          # B[0, k0]
        v_f = k1          # B[0, k1]
        v_g = p + k0      # B[1, k0]
        v_h = p + k1      # B[1, k1]

        w_00 = k0 * n     # C[0, k0]
        w_01 = k1 * n     # C[0, k1]
        w_10 = k0 * n + 1 # C[1, k0]
        w_11 = k1 * n + 1 # C[1, k1]

        # M1 = (a+d)(e+h)
        U[0, col] += 1
        U[4, col] += 1
        V[v_e, col] += 1
        V[v_h, col] += 1

        # M2 = (c+d)e
        U[3, col + 1] += 1
        U[4, col + 1] += 1
        V[v_e, col + 1] += 1

        # M3 = a(f-h)
        U[0, col + 2] += 1
        V[v_f, col + 2] += 1
        V[v_h, col + 2] -= 1

        # M4 = d(g-e)
        U[4, col + 3] += 1
        V[v_g, col + 3] += 1
        V[v_e, col + 3] -= 1

        # M5 = (a+b)h
        U[0, col + 4] += 1
        U[1, col + 4] += 1
        V[v_h, col + 4] += 1

        # M6 = (c-a)(e+f)
        U[3, col + 5] += 1
        U[0, col + 5] -= 1
        V[v_e, col + 5] += 1
        V[v_f, col + 5] += 1

        # M7 = (b-d)(g+h)
        U[1, col + 6] += 1
        U[4, col + 6] -= 1
        V[v_g, col + 6] += 1
        V[v_h, col + 6] += 1

        # Reconstruct C[0,k0] = M1 + M4 - M5 + M7
        W[w_00, col] += 1
        W[w_00, col + 3] += 1
        W[w_00, col + 4] -= 1
        W[w_00, col + 6] += 1

        # C[0,k1] = M3 + M5
        W[w_01, col + 2] += 1
        W[w_01, col + 4] += 1

        # C[1,k0] = M2 + M4
        W[w_10, col + 1] += 1
        W[w_10, col + 3] += 1

        # C[1,k1] = M1 - M2 + M3 + M6
        W[w_11, col] += 1
        W[w_11, col + 1] -= 1
        W[w_11, col + 2] += 1
        W[w_11, col + 5] += 1

        col += 7

    # Part 2: Naive for B col 8, rows 0-1 (2x2 @ 2x1)
    k8 = 8
    v_e8 = k8          # B[0, 8]
    v_g8 = p + k8      # B[1, 8]
    w_08 = k8 * n      # C[0, 8]
    w_18 = k8 * n + 1  # C[1, 8]

    U[0, col] += 1
    V[v_e8, col] += 1
    W[w_08, col] += 1

    U[1, col + 1] += 1
    V[v_g8, col + 1] += 1
    W[w_08, col + 1] += 1

    U[3, col + 2] += 1
    V[v_e8, col + 2] += 1
    W[w_18, col + 2] += 1

    U[4, col + 3] += 1
    V[v_g8, col + 3] += 1
    W[w_18, col + 3] += 1

    col += 4

    # Part 3: A[:,2] @ B[2,:] (2x1 @ 1x9)
    u_ar0 = 2  # A[0,2]
    u_ar1 = 5  # A[1,2]

    for k in range(p):
        v_bk = 2 * p + k  # B[2, k]
        w_0k = k * n      # C[0, k]
        w_1k = k * n + 1  # C[1, k]

        U[u_ar0, col] += 1
        V[v_bk, col] += 1
        W[w_0k, col] += 1

        U[u_ar1, col + 1] += 1
        V[v_bk, col + 1] += 1
        W[w_1k, col + 1] += 1

        col += 2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
