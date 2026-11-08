<<<<<<< SEARCH
def construct():
    # Guaranteed fallback
    fallback = _build_rank65()
    # Attempt to find a better 4x4x4 decomposition
    best = _try_find_better()
    if best is not None:
        r44, U44, V44, W44 = best
        # Build full 4x4x5: first 4 cols use 4x4x4 decomp, 5th col schoolbook (16)
        n, m, p = 4, 4, 5
        rank = r44 + 16
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for t in range(r44):
            for i in range(n*m):
                U[i][t] = U44[i][t]
            for j in range(m*p):
                V[j][t] = V44[j][t]
            for k in range(p*n):
                W[k][t] = W44[k][t]
        col = r44
        for i in range(4):
            for j in range(4):
                U[i*m+j][col] += 1
                V[j*p+4][col] += 1
                W[4*n+i][col] += 1
                col += 1
        result = {"U":U, "V":V, "W":W}
        if _verify(U, V, W):
            return result
    return fallback
=======
def construct():
    # Guaranteed fallback
    fallback = _build_rank65()
    # Attempt to find a better 4x4x4 decomposition
    best = _try_find_better()
    if best is not None:
        r44, U44, V44, W44 = best
        # Build full 4x4x5: first 4 cols use 4x4x4 decomp, 5th col schoolbook (16)
        n, m, p = 4, 4, 5
        rank = r44 + 16
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        for t in range(r44):
            for i in range(n*m):
                U[i][t] = U44[i][t]
            for j in range(m*p):
                V[j][t] = V44[j][t]
            for k in range(p*n):
                W[k][t] = W44[k][t]
        col = r44
        for i in range(4):
            for j in range(4):
                U[i*m+j][col] += 1
                V[j*p+4][col] += 1
                W[4*n+i][col] += 1
                col += 1
        result = {"U":U, "V":V, "W":W}
        if _verify(U, V, W):
            return result
    # Also try: direct 4x4x5 search with rank < 65
    # The 4x4x5 tensor has 4*4*5=80 entries. Rank lower bound is max(4,4,5)=5 but
    # for MM it's much higher. Try CP-ALS directly on the 4x4x5 tensor.
    if _time_left() > 15:
        n, m, p = 4, 4, 5
        T5 = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if i == k:
                        T5[i, j, k] = 1
        # Try ranks from 20 to 64
        for r in range(20, 65):
            if _time_left() < 10:
                break
            for seed in range(4):
                if _time_left() < 5:
                    break
                Uf, Vf, Wf = _fast_cp_als(T5, r, seed, n_iter=80)
                # Round to integers
                Ui = np.rint(Uf)
                Vi = np.rint(Vf)
                Wi = np.rint(Wf)
                # Verify 4x4x5
                Tcheck = np.zeros((n, m, p), dtype=object)
                for t in range(r):
                    u = Ui[:, t]
                    v = Vi[:, t]
                    w = Wi[:, t]
                    for i in range(n):
                        for j in range(m):
                            uij = u[i*m + j]
                            if uij == 0:
                                continue
                            for k in range(p):
                                vjk = v[j*p + k]
                                if vjk == 0:
                                    continue
                                wki = w[k*n + i]
                                if wki == 0:
                                    continue
                                Tcheck[i, j, k] += uij * vjk * wki
                ok = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            expected = 1 if i == k else 0
                            if Tcheck[i, j, k] != expected:
                                ok = False
                                break
                        if not ok:
                            break
                    if not ok:
                        break
                if ok:
                    Uout = Ui.tolist()
                    Vout = Vi.tolist()
                    Wout = Wi.tolist()
                    return {"U": Uout, "V": Vout, "W": Wout}
    return fallback
>>>>>>> REPLACE
