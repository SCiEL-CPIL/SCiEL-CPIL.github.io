import numpy as np
import time

DIMENSIONS = [2, 4, 7]


def construct():
    start_time = time.time()
    deadline = start_time + 148

    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n

    def matmul_tensor():
        T = np.zeros((N, M, P), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T_target = matmul_tensor()

    def check_decomp(U, V, W):
        T = np.einsum('ir,jr,kr->ijk', U, V, W)
        return np.max(np.abs(T - T_target))

    # Start with known-good block decomposition (2+2 on m, 2+2+2+1 on p)
    U_list, V_list, W_list = [], [], []

    def add_block(i0, j0, k0, nn, mm, pp):
        # Block: rows i0..i0+nn-1, cols j0..j0+mm-1, output cols k0..k0+pp-1
        if nn == 2 and mm == 2 and pp == 2:
            # Strassen 7
            a, b = i0, i0 + 1
            c, d = j0, j0 + 1
            e, f = k0, k0 + 1
            for (ui, vj, wk) in [
                ((a, c), (c, e), (e, a)),  # S1 = (A11+A22)(B11+B22) -> C11, C22
            ]:
                pass
            # Standard Strassen for 2x2x2 with indices (a,b) rows, (c,d) cols, (e,f) out
            # S1 = (A11+A22)(B11+B22): C11 += S1, C22 += S1
            s1 = U_list[-1] if U_list else None
            # We'll build the 7 terms directly
            terms = [
                ({(a, c), (a, d)}, {(c, e), (d, f)}, {(e, a), (f, a)}),  # M1=(A11+A22)(B11+B22)
                ({(b, c), (b, d)}, {(c, e)}, {(e, b), (f, b)}),          # M2=(A21+A22)B11
                ({(a, d)}, {(d, f)}, {(f, a)}),                           # M3=A12(B22-B11)
                ({(b, d)}, {(c, f), (d, f)}, {(e, a), (e, b)}),           # M4=A22(B12+B21)
                ({(a, c), (b, c)}, {(d, f)}, {(e, a), (e, b)}),           # M5=(A11-A22)(B12+B22)
                ({(b, c), (a, c)}, {(c, e), (d, f)}, {(f, a)}),           # M6=(A21-A11)(B11+B12)
                ({(b, d), (a, d)}, {(c, f), (d, f)}, {(e, b)}),           # M7=(A12-A22)(B21+B22)
            ]
            # Hmm, let me just use the existing verified pattern
            pass

    # Just use the verified 50-rank construction directly
    U = [np.zeros(50) for _ in range(N)]
    V = [np.zeros(50) for _ in range(M)]
    W = [np.zeros(50) for _ in range(P)]
    col = 0

    def add_strassen_222(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)

        U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1; W[c00][col] += 1; W[c11][col] += 1; col += 1
        U[c][col] += 1; U[d][col] += 1; V[e][col] += 1; W[c10][col] += 1; W[c11][col] -= 1; col += 1
        U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1; W[c01][col] += 1; W[c11][col] += 1; col += 1
        U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1; W[c00][col] += 1; W[c10][col] += 1; col += 1
        U[a][col] += 1; U[b][col] += 1; V[h][col] += 1; W[c00][col] -= 1; W[c01][col] += 1; col += 1
        U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1; W[c11][col] += 1; col += 1
        U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1; W[c00][col] += 1; col += 1
        return True

    def add_schoolbook_221(i0, j0, k0):
        nonlocal col
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        g = (j0 + 1) * p + k0
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)

        U[a][col] += 1; V[e][col] += 1; W[c00][col] += 1; col += 1
        U[b][col] += 1; V[g][col] += 1; W[c00][col] += 1; col += 1
        U[c][col] += 1; V[e][col] += 1; W[c10][col] += 1; col += 1
        U[d][col] += 1; V[g][col] += 1; W[c10][col] += 1; col += 1
        return True

    def add_schoolbook_223(i0, j0, k0):
        nonlocal col
        # 2x2x3 block via 6 schoolbook terms
        for di in range(2):
            for dj in range(2):
                for dk in range(3):
                    if col >= 50:
                        return False
                    a = (i0 + di) * m + (j0 + dj)
                    b = (j0 + dj) * p + (k0 + dk)
                    c = (k0 + dk) * n + (i0 + di)
                    U[a][col] += 1; V[b][col] += 1; W[c][col] += 1; col += 1
        return True

    # Original: 6 strassen + 2 schoolbook_221 = 50
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 0, k0):
            break
    if not add_schoolbook_221(0, 0, 6):
        pass
    for k0 in [0, 2, 4]:
        if not add_strassen_222(0, 2, k0):
            break
    if not add_schoolbook_221(0, 2, 6):
        pass

    # Convert to plain lists for return
    U_out = [[float(x) for x in row] for row in U]
    V_out = [[float(x) for x in row] for row in V]
    W_out = [[float(x) for x in row] for row in W]

    # Trim trailing all-zero rank columns
    while col < 50:
        if all(U[i][col] == 0 for i in range(N)) and \
           all(V[i][col] == 0 for i in range(M)) and \
           all(W[i][col] == 0 for i in range(P)):
            for i in range(N):
                U_out[i].pop()
            for i in range(M):
                V_out[i].pop()
            for i in range(P):
                W_out[i].pop()
            col -= 1
        else:
            break
    col = 50

    return {"U": U_out, "V": V_out, "W": W_out}
