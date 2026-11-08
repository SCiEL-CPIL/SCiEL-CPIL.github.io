import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    rank = 52
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Tensor indexing:
    # A: i*m + j  (i in [0,1], j in [0,2])
    # B: j*p + k  (j in [0,2], k in [0,8])
    # C: k*n + i  (k in [0,8], i in [0,1])

    col = 0

    # === Block 1: A[:, 0:2] (2x2) * B[0:2, :] (2x9) ===
    # Strassen on 4 pairs of B columns: (0,1), (2,3), (4,5), (6,7)
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1

        v_e = 0 * p + k0
        v_f = 0 * p + k1
        v_g = 1 * p + k0
        v_h = 1 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # 7 Strassen products
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        U[u_c][col + 1] += 1
        U[u_d][col + 1] += 1
        V[v_e][col + 1] += 1
        U[u_a][col + 2] += 1
        V[v_f][col + 2] += 1
        V[v_h][col + 2] -= 1
        U[u_d][col + 3] += 1
        V[v_g][col + 3] += 1
        V[v_e][col + 3] -= 1
        U[u_a][col + 4] += 1
        U[u_b][col + 4] += 1
        V[v_h][col + 4] += 1
        U[u_c][col + 5] += 1
        U[u_a][col + 5] -= 1
        V[v_e][col + 5] += 1
        V[v_f][col + 5] += 1
        U[u_b][col + 6] += 1
        U[u_d][col + 6] -= 1
        V[v_g][col + 6] += 1
        V[v_h][col + 6] += 1

        # Reconstruct outputs:
        W[w_00][col] += 1
        W[w_00][col + 3] += 1
        W[w_00][col + 4] -= 1
        W[w_00][col + 6] += 1
        W[w_01][col + 2] += 1
        W[w_01][col + 4] += 1
        W[w_10][col + 1] += 1
        W[w_10][col + 3] += 1
        W[w_11][col] += 1
        W[w_11][col + 1] -= 1
        W[w_11][col + 2] += 1
        W[w_11][col + 5] += 1

        col += 7

    # Handle column k=8 for A[:, 0:2] * B[0:2, 8]
    k8 = 8
    u_a = 0 * m + 0
    u_b = 0 * m + 1
    u_c = 1 * m + 0
    u_d = 1 * m + 1
    v_e = 0 * p + k8
    v_g = 1 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1

    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col + 1] += 1
    V[v_g][col + 1] += 1
    U[u_c][col + 2] += 1
    V[v_e][col + 2] += 1
    U[u_d][col + 3] += 1
    V[v_g][col + 3] += 1

    W[w_08][col] += 1
    W[w_08][col + 1] += 1
    W[w_18][col + 2] += 1
    W[w_18][col + 3] += 1

    col += 4

    # === Block 2: A[:, 2] (2x1) * B[2, :] (1x9) ===
    # Instead of 2 rank-1 terms per column, use Strassen-like trick:
    # For each pair of columns (k0, k1), use 3 rank-1 terms instead of 4
    u_ar0 = 0 * m + 2
    u_ar1 = 1 * m + 2

    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = 2 * p + k0
        v_f = 2 * p + k1
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1 = a0 * e, M2 = a1 * f, M3 = (a0+a1)*(e+f)
        # C[0,k0] = M1, C[0,k1] = M2, C[1,k0] = M1 - M2 + M3, C[1,k1] = M2 - M1 + M3
        # Wait, that doesn't work directly. Let me think again.
        # Actually for 2x1 * 1x2:
        # C = [[a0*e, a0*f], [a1*e, a1*f]]
        # This is 4 independent entries, needs 4 rank-1 terms.
        # But if we allow combining: use (a0+a1)*(e+f) to get a0*e+a0*f+a1*e+a1*f
        # Then C[0,k0] = a0*e = ?
        # This doesn't reduce rank. Let's just use 4 terms.
        U[u_ar0][col] += 1
        V[v_e][col] += 1
        W[w_00][col] += 1
        U[u_ar0][col + 1] += 1
        V[v_f][col + 1] += 1
        W[w_01][col + 1] += 1
        U[u_ar1][col + 2] += 1
        V[v_e][col + 2] += 1
        W[w_10][col + 2] += 1
        U[u_ar1][col + 3] += 1
        V[v_f][col + 3] += 1
        W[w_11][col + 3] += 1
        col += 4

    # Last column k=8
    k8 = 8
    v_e = 2 * p + k8
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    U[u_ar0][col] += 1
    V[v_e][col] += 1
    W[w_08][col] += 1
    U[u_ar1][col + 1] += 1
    V[v_e][col + 1] += 1
    W[w_18][col + 1] += 1
    col += 2

    return {"U": U, "V": V, "W": W}
