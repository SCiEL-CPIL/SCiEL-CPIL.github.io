import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    # We use a rank-50 exact decomposition for (2,3,9).
    # Structure: split A into left 2x2 block and right 2x1 block, and
    # use Strassen on the 4 (2x2)x(2x9) sub-problems plus direct products
    # for the remaining terms.
    rank = 50
    U = [[0.0] * rank for _ in range(n * m)]   # 6 x 50
    V = [[0.0] * rank for _ in range(m * p)]   # 27 x 50
    W = [[0.0] * rank for _ in range(p * n)]   # 18 x 50

    # Index helpers
    def aidx(i, j): return i * m + j
    def bidx(j, k): return j * p + k
    def cidx(k, i): return k * n + i

    col = 0

    # ---- Part 1: A_left (2x2) * B_top (2x9) using 4 Strassen 7-product blocks ----
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = bidx(0, k0)
        v_f = bidx(0, k1)
        v_g = bidx(1, k0)
        v_h = bidx(1, k1)
        u_a = aidx(0, 0)
        u_b = aidx(0, 1)
        u_c = aidx(1, 0)
        u_d = aidx(1, 1)

        # M1 = (a+d)(e+h)
        U[u_a][col] += 1.0; U[u_d][col] += 1.0
        V[v_e][col] += 1.0; V[v_h][col] += 1.0
        # M2 = (c+d)e
        U[u_c][col+1] += 1.0; U[u_d][col+1] += 1.0
        V[v_e][col+1] += 1.0
        # M3 = a(f-h)
        U[u_a][col+2] += 1.0
        V[v_f][col+2] += 1.0; V[v_h][col+2] -= 1.0
        # M4 = d(g-e)
        U[u_d][col+3] += 1.0
        V[v_g][col+3] += 1.0; V[v_e][col+3] -= 1.0
        # M5 = (a+b)h
        U[u_a][col+4] += 1.0; U[u_b][col+4] += 1.0
        V[v_h][col+4] += 1.0
        # M6 = (c-a)(e+f)
        U[u_c][col+5] += 1.0; U[u_a][col+5] -= 1.0
        V[v_e][col+5] += 1.0; V[v_f][col+5] += 1.0
        # M7 = (b-d)(g+h)
        U[u_b][col+6] += 1.0; U[u_d][col+6] -= 1.0
        V[v_g][col+6] += 1.0; V[v_h][col+6] += 1.0

        w_00 = cidx(k0, 0)
        w_01 = cidx(k1, 0)
        w_10 = cidx(k0, 1)
        w_11 = cidx(k1, 1)

        # C_{0,k0} = M1 + M4 - M5 + M7
        W[w_00][col]   += 1.0
        W[w_00][col+3] += 1.0
        W[w_00][col+4] -= 1.0
        W[w_00][col+6] += 1.0
        # C_{0,k1} = M3 + M5
        W[w_01][col+2] += 1.0
        W[w_01][col+4] += 1.0
        # C_{1,k0} = M2 + M4
        W[w_10][col+1] += 1.0
        W[w_10][col+3] += 1.0
        # C_{1,k1} = M1 - M2 + M3 + M6
        W[w_11][col]   += 1.0
        W[w_11][col+1] -= 1.0
        W[w_11][col+2] += 1.0
        W[w_11][col+5] += 1.0

        col += 7

    # Last column k=8 handled with 4 direct products (2x1 * 2x1)
    k8 = 8
    v_e = bidx(0, k8)
    v_g = bidx(1, k8)
    u_a = aidx(0, 0); u_b = aidx(0, 1); u_c = aidx(1, 0); u_d = aidx(1, 1)

    U[u_a][col]   += 1.0; V[v_e][col]   += 1.0   # P1 = a*e
    U[u_b][col+1] += 1.0; V[v_g][col+1] += 1.0   # P2 = b*g
    U[u_c][col+2] += 1.0; V[v_e][col+2] += 1.0   # P3 = c*e
    U[u_d][col+3] += 1.0; V[v_g][col+3] += 1.0   # P4 = d*g

    w_08 = cidx(k8, 0)
    w_18 = cidx(k8, 1)
    W[w_08][col]   += 1.0; W[w_08][col+1] += 1.0
    W[w_18][col+2] += 1.0; W[w_18][col+3] += 1.0
    col += 4

    # ---- Part 2: A_right (2x1) * B_bottom (1x9) direct products ----
    u_ar0 = aidx(0, 2)
    u_ar1 = aidx(1, 2)
    for k in range(9):
        v_b = bidx(2, k)
        U[u_ar0][col]   += 1.0; V[v_b][col]   += 1.0
        W[cidx(k, 0)][col] += 1.0
        U[u_ar1][col+1] += 1.0; V[v_b][col+1] += 1.0
        W[cidx(k, 1)][col+1] += 1.0
        col += 2

    # Convert to lists of ints/floats (integer values here)
    U = [[int(v) for v in row] for row in U]
    V = [[int(v) for v in row] for row in V]
    W = [[int(v) for v in row] for row in W]

    return {"U": U, "V": V, "W": W}
