import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 140.0
    n, m, p = 4, 4, 4

    def build_strassen_factors(size):
        if size == 1:
            return ([(0, 0)], [(0, 0)], [(0, 0)])
        half = size // 2
        L, R, C = build_strassen_factors(half)
        # 7 Strassen sub-products, each with (L_dict, R_dict, C_dict)
        subs = [
            ([(0, 0)], [(0, 0)], [(0, 0), (1, 1)]),
            ([(1, 0), (1, 1)], [(0, 0)], [(1, 0), (1, 1)]),
            ([(0, 0)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),
            ([(1, 1)], [(1, 0), (0, 0)], [(0, 0), (1, 0)]),
            ([(0, 0), (0, 1)], [(1, 1)], [(0, 0), (0, 1)]),
            ([(1, 0), (0, 0)], [(0, 0), (0, 1)], [(1, 1)]),
            ([(0, 1), (1, 1)], [(1, 0), (1, 1)], [(0, 0)]),
        ]
        signs = [
            {(-1, 1): -1, (0, 1): -1, (1, 0): -1, (1, 1): 1},
            {(1, 1): -1},
            {(1, 1): -1},
            {(0, 0): -1},
            {(0, 0): -1},
            {(0, 0): -1},
            {(1, 1): -1},
        ]
        U, V, W = [], [], []
        for si in range(7):
            Ls, Rs, Cs = subs[si]
            sign_map = signs[si]
            U_d, V_d, W_d = {}, {}, {}
            for (br, bc) in Ls:
                for (lr, lc), coeff in L:
                    key = (br * half + lr, bc * half + lc)
                    U_d[key] = U_d.get(key, 0) + coeff
            for (br, bc) in Rs:
                for (lr, lc), coeff in R:
                    key = (br * half + lr, bc * half + lc)
                    V_d[key] = V_d.get(key, 0) + coeff
            for (br, bc) in Cs:
                for (lr, lc), coeff in C:
                    key = (br * half + lr, bc * half + lc)
                    base = W_d.get(key, 0)
                    W_d[key] = base + sign_map.get((br, bc), 1) * coeff
            U.append(U_d); V.append(V_d); W.append(W_d)
        return (U, V, W)

    U_list, V_list, W_list = build_strassen_factors(n)
    rank = len(U_list)
    N = n * m
    U = [[0] * rank for _ in range(N)]
    V = [[0] * rank for _ in range(N)]
    W = [[0] * rank for _ in range(N)]
    for r in range(rank):
        for (i, k), v in U_list[r].items():
            U[i * m + k][r] = v
        for (k, j), v in V_list[r].items():
            V[k * p + j][r] = v
        for (i, j), v in W_list[r].items():
            W[i * n + j][r] = v

    def factor_matrix(M, r):
        return np.array([M[row][r] for row in range(N)], dtype=float)

    def validate(keep):
        T = np.zeros((N, N, N), dtype=float)
        for r in keep:
            T += np.einsum('a,b,c->abc', factor_matrix(U, r), factor_matrix(V, r), factor_matrix(W, r))
        for i in range(n):
            for k in range(m):
                for j in range(p):
                    if abs(T[i * m + k, k * p + j, k * n + i] - 1) > 1e-9:
                        return False
        for a in range(N):
            for b in range(N):
                for c in range(N):
                    if a // m == b // p and a // m == c // n and b // p == c // n:
                        continue
                    if abs(T[a, b, c]) > 1e-9:
                        return False
        return True

    if not validate(list(range(rank))):
        return {"U": U, "V": V, "W": W}

    # Greedy rank reduction: try to drop each term, validate, repeat
    keep = list(range(rank))
    for _ in range(30):
        if time.time() > deadline - 20:
            break
        removed = False
        for r in list(keep):
            if time.time() > deadline - 20:
                break
            new_keep = [x for x in keep if x != r]
            if validate(new_keep):
                keep = new_keep
                removed = True
                break
        if not removed:
            break

    if len(keep) < rank:
        U = [row[i] for row in U] if False else [
            [row[r] for r in keep] for row in U
        ]
        V = [[row[r] for r in keep] for row in V]
        W = [[row[r] for r in keep] for row in W]

    return {"U": U, "V": V, "W": W}
