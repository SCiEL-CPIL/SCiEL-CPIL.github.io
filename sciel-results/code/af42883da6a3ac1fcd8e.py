"""Tensor decomposition for 4x4 * 4x6 matrix multiplication.
Rank 84 via three 4x4x2 subproblems, each = four rank-7 Strassen 2x2x2 products."""
import time
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under the 120s wall-clock limit

    n, m, p = DIMENSIONS  # 4, 4, 6

    def build_strassen():
        cols_U, cols_V, cols_W = [], [], []

        def add_term(u_fac, v_fac, w_fac):
            cols_U.append(dict(u_fac))
            cols_V.append(dict(v_fac))
            cols_W.append(dict(w_fac))

        X = lambda a, b: a * 2 + b
        Y = lambda a, b: a * 2 + b
        Z = lambda a, b: a * 2 + b
        # Strassen 2x2x2: seven rank-1 terms (x_factor, y_factor, z_factor).
        terms = [
            ({X(0,0):1, X(1,1):1},   {Y(0,0):1, Y(1,1):1},     {Z(0,0):1,  Z(1,1):1}),   # m1
            ({X(1,0):1, X(1,1):1},   {Y(0,0):1},               {Z(1,0):1,  Z(1,1):1}),   # m2
            ({X(0,0):1},             {Y(0,1):1, Y(1,1):-1},    {Z(0,1):1,  Z(1,1):-1}),  # m3
            ({X(1,1):1},             {Y(1,0):1, Y(0,0):-1},    {Z(0,0):-1, Z(1,0):1}),   # m4
            ({X(0,0):1, X(0,1):1},   {Y(1,1):1},               {Z(0,0):1,  Z(0,1):1}),   # m5
            ({X(1,0):1, X(0,0):-1},  {Y(0,0):1, Y(0,1):1},     {Z(0,0):1,  Z(1,0):1}),   # m6
            ({X(0,1):1, X(1,1):-1},  {Y(1,0):1, Y(1,1):1},     {Z(1,1):1}),              # m7
        ]

        # Three 4x4x2 subproblems (B/C column pairs 2s,2s+1).
        for s in range(3):
            if time.time() > deadline:
                break
            # Four 2x2x2 Strassen products per subproblem:
            # C_top = A11*B_top + A12*B_bot ; C_bot = A21*B_top + A22*B_bot
            for (r, c, rb) in [(0, 0, 0), (0, 1, 1), (1, 0, 0), (1, 1, 1)]:
                for (xf, yf, zf) in terms:
                    u_fac = {}
                    for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        idx = a * 2 + b
                        if idx in xf:
                            u_fac[(2 * r + a) * m + (2 * c + b)] = xf[idx]
                    v_fac = {}
                    for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        idx = a * 2 + b
                        if idx in yf:
                            v_fac[(2 * rb + a) * p + (2 * s + b)] = yf[idx]
                    w_fac = {}
                    for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                        idx = a * 2 + b
                        if idx in zf:
                            w_fac[(2 * s + b) * n + (2 * r + a)] = zf[idx]
                    add_term(u_fac, v_fac, w_fac)

        rank = len(cols_U)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        for c in range(rank):
            for (row, val) in cols_U[c].items():
                U[row][c] = val
            for (row, val) in cols_V[c].items():
                V[row][c] = val
            for (row, val) in cols_W[c].items():
                W[row][c] = val
        return {"U": U, "V": V, "W": W}

    def build_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify(res):
        try:
            rank = len(res["U"][0])
            U3 = np.array([[res["U"][i * m + j][c] for c in range(rank)]
                           for i in range(n) for j in range(m)]).reshape(n, m, rank)
            V3 = np.array([[res["V"][j * p + k][c] for c in range(rank)]
                           for j in range(m) for k in range(p)]).reshape(m, p, rank)
            W3 = np.array([[res["W"][k * n + i][c] for c in range(rank)]
                           for k in range(p) for i in range(n)]).reshape(p, n, rank)
            T = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
            return bool(np.allclose(T, 1.0, atol=1e-9))
        except Exception:
            return False

    res = build_strassen()
    if not verify(res):
        res = build_schoolbook()
    return res
# EVOLVE-BLOCK-END
