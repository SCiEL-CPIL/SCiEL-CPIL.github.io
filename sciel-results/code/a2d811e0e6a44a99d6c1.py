"""Tensor decomposition for 2x3 * 3x10 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 2, 3, 10
    rank = 55
    U = [[0] * rank for _ in range(n * m)]   # 6 x 55
    V = [[0] * rank for _ in range(m * p)]   # 30 x 55
    W = [[0] * rank for _ in range(p * n)]   # 20 x 55

    r = 0

    # Strassen 7 products for 2x2 x 2x2:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # m1=(a+d)(e+h), m2=(c+d)e, m3=a(f-h), m4=d(g-e),
    # m5=(a+b)h, m6=(c-a)(e+f), m7=(b-d)(g+h)
    # C00=m1+m4-m5+m7, C01=m3+m5, C10=m2+m4, C11=m1-m2+m3+m6

    # U offsets: a=row0, b=row1, c=row3, d=row4
    strassen_u = [
        [(0, 1), (4, 1)],      # m1: a+d
        [(3, 1), (4, 1)],      # m2: c+d
        [(0, 1)],              # m3: a
        [(4, 1)],              # m4: d
        [(0, 1), (1, 1)],      # m5: a+b
        [(3, 1), (0, -1)],     # m6: c-a
        [(1, 1), (4, -1)],     # m7: b-d
    ]

    # V offsets: e=row0, f=row1, g=row10, h=row11
    strassen_v = [
        [(0, 1), (11, 1)],     # m1: e+h
        [(0, 1)],              # m2: e
        [(1, 1), (11, -1)],    # m3: f-h
        [(10, 1), (0, -1)],    # m4: g-e
        [(11, 1)],             # m5: h
        [(0, 1), (1, 1)],      # m6: e+f
        [(10, 1), (11, 1)],    # m7: g+h
    ]

    # W offsets: 0=C[0,col0], 1=C[1,col0], 2=C[0,col1], 3=C[1,col1]
    strassen_w = [
        [(0, 1), (3, 1)],      # m1: C00 + C11
        [(1, 1), (3, -1)],     # m2: C10 - C11
        [(2, 1), (3, 1)],      # m3: C01 + C11
        [(0, 1), (1, 1)],      # m4: C00 + C10
        [(0, -1), (2, 1)],     # m5: -C00 + C01
        [(3, 1)],              # m6: C11
        [(0, 1)],              # m7: C00
    ]

    # Apply Strassen to 5 disjoint 2x2 blocks (columns 0-9 of B)
    for b in range(5):
        col0 = 2 * b
        col1 = 2 * b + 1

        for s in range(7):
            for (u_off, coeff) in strassen_u[s]:
                U[u_off][r] += coeff

            for (v_off, coeff) in strassen_v[s]:
                if v_off == 0:
                    v_row = col0
                elif v_off == 1:
                    v_row = col1
                elif v_off == 10:
                    v_row = 10 + col0
                elif v_off == 11:
                    v_row = 10 + col1
                else:
                    raise ValueError(f"Bad v_off {v_off}")
                V[v_row][r] += coeff

            for (w_off, coeff) in strassen_w[s]:
                if w_off == 0:
                    w_row = col0 * 2 + 0
                elif w_off == 1:
                    w_row = col0 * 2 + 1
                elif w_off == 2:
                    w_row = col1 * 2 + 0
                elif w_off == 3:
                    w_row = col1 * 2 + 1
                else:
                    raise ValueError(f"Bad w_off {w_off}")
                W[w_row][r] += coeff

            r += 1

    # Schoolbook for A_right (2x1) @ B_bottom (1x10): 2*10 = 20 terms
    for i in range(2):
        for k in range(10):
            U[i * 3 + 2][r] = 1
            V[20 + k][r] = 1
            W[k * 2 + i][r] = 1
            r += 1

    # Validate: sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                if not np.isclose(lhs, 1.0, atol=1e-9):
                    # Fallback to trivial rank-60
                    rank60 = 60
                    U2 = [[0] * rank60 for _ in range(n * m)]
                    V2 = [[0] * rank60 for _ in range(m * p)]
                    W2 = [[0] * rank60 for _ in range(p * n)]
                    for rr in range(rank60):
                        ii = rr // (m * p)
                        rem = rr % (m * p)
                        jj = rem // p
                        kk = rem % p
                        U2[ii * m + jj][rr] = 1
                        V2[jj * p + kk][rr] = 1
                        W2[kk * n + ii][rr] = 1
                    return {"U": U2, "V": V2, "W": W2}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
