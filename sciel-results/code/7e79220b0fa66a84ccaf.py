import numpy as np

def construct():
    n, m, p = 2, 2, 14
    num_blocks = p // 2

    # Strassen terms: (a_mask, b_mask, c_mask)
    # Each mask is a 4-element tuple for a 2x2 matrix in row-major order
    # Term s: U_part = a_mask * A, V_part = b_mask * B, W_part = c_mask * C
    # where (a_mask * A) * (b_mask * B) contributes c_mask to C
    strassen = [
        ((1, 0, 0, 1), (1, 0, 0, 1), (1, 0, 0, 1)),
        ((0, 0, 1, 1), (1, 0, 0, 0), (0, 0, 1, -1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
        ((0, 0, 0, 1), (0, 0, 1, -1), (1, 0, 1, 0)),
        ((1, 1, 0, 0), (0, 0, 0, 1), (-1, 1, 0, 0)),
        ((0, 0, 1, -1), (1, 1, 0, 0), (0, 0, 0, 1)),
        ((0, 1, 0, -1), (0, 0, 1, 1), (1, 0, 0, 0)),
    ]

    rank = 7 * num_blocks
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for t in range(num_blocks):
        for s in range(7):
            r = 7 * t + s
            am, bm, cm = strassen[s]
            for a in range(4):
                if am[a] != 0:
                    U[a, r] += am[a]
            for b in range(4):
                if bm[b] != 0:
                    V[b + 2 * t, r] += bm[b]
            for c in range(4):
                if cm[c] != 0:
                    W[c + 2 * t, r] += cm[c]

    # Verify the decomposition
    ok = True
    for i in range(n):
        for j in range(m):
            ui = i * m + j
            for k in range(p):
                vi = j * p + k
                wi = k * n + i
                val = np.dot(U[ui], V[vi] * W[wi])
                if val != 1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break

    if not ok:
        # Fallback: schoolbook rank-56
        U = np.zeros((n * m, n * m * p), dtype=int)
        V = np.zeros((m * p, n * m * p), dtype=int)
        W = np.zeros((p * n, n * m * p), dtype=int)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
                    c += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
