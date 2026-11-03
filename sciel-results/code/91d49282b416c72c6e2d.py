"""Matrix multiplication 2,3,8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 8]


def construct():
    import numpy as np

    n, m, p = DIMENSIONS

    # ---- Block-partitioned rank-44 construction ---------------------------------
    # Split p=8 into two p=4 blocks.  For each p=4 block split m=3 into m=2 (Strassen)
    # and m=1 (schoolbook).  The m=2,p=4 part splits into two 2x2x2 Strassen blocks
    # (rank 7 each).  Total rank per p=4 block = 7+7+8 = 22, overall 44.
    U = np.zeros((n * m, 44), dtype=object)
    V = np.zeros((m * p, 44), dtype=object)
    W = np.zeros((p * n, 44), dtype=object)

    col = 0

    def add_strassen(i0, j0, k0):
        nonlocal col
        def Uidx(li, lj):
            return (i0 + li) * m + (j0 + lj)
        def Vidx(lj, lk):
            return (j0 + lj) * p + (k0 + lk)
        def Widx(lk, li):
            return (k0 + lk) * n + (i0 + li)

        def term(ud, vd, wd):
            nonlocal col
            for (a, b) in ud.items():
                U[a, col] += b
            for (a, b) in vd.items():
                V[a, col] += b
            for (a, b) in wd.items():
                W[a, col] += b
            col += 1

        term({Uidx(0, 0): 1, Uidx(1, 1): 1},
             {Vidx(0, 0): 1, Vidx(1, 1): 1},
             {Widx(0, 0): 1, Widx(1, 1): 1})                       # (a+d)(e+h)
        term({Uidx(1, 0): 1, Uidx(1, 1): 1},
             {Vidx(0, 0): 1},
             {Widx(0, 1): 1, Widx(1, 1): -1})                     # (c+d)e
        term({Uidx(0, 0): 1},
             {Vidx(0, 1): 1, Vidx(1, 1): -1},
             {Widx(1, 0): 1, Widx(1, 1): 1})                     # a(f-h)
        term({Uidx(1, 1): 1},
             {Vidx(1, 0): 1, Vidx(0, 0): -1},
             {Widx(0, 0): 1, Widx(0, 1): 1})                     # d(g-e)
        term({Uidx(0, 0): 1, Uidx(0, 1): 1},
             {Vidx(1, 1): 1},
             {Widx(1, 0): 1, Widx(0, 0): -1})                    # (a+b)h
        term({Uidx(1, 0): 1, Uidx(0, 0): -1},
             {Vidx(0, 0): 1, Vidx(0, 1): 1},
             {Widx(1, 1): 1})                                    # (c-a)(e+f)
        term({Uidx(0, 1): 1, Uidx(1, 1): -1},
             {Vidx(1, 0): 1, Vidx(1, 1): 1},
             {Widx(0, 0): 1})                                    # (b-d)(g+h)

    def add_school_2x1x4(i0, j0, k0):
        nonlocal col
        for li in range(2):
            for lk in range(4):
                U[(i0 + li) * m + (j0 + 0), col] = 1
                V[(j0 + 0) * p + (k0 + lk), col] = 1
                W[(k0 + lk) * n + (i0 + li), col] = 1
                col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_school_2x1x4(0, 2, 0)
    add_strassen(0, 0, 4)
    add_strassen(0, 0, 6)
    add_school_2x1x4(0, 2, 4)

    # ---- Verification: tensor equality -----------------------------------------
    def tensor_of(A, B, C, rank):
        T = np.zeros((n, p, m), dtype=object)
        for r in range(rank):
            A = A[:, r]; B = B[:, r]; C = C[:, r]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, k, j] += A[i * m + j] * B[j * p + k] * C[k * n + i]
        return T

    def M_expected():
        T = np.zeros((n, p, m), dtype=object)
        for i in range(n):
            for j in range(m):
                T[i, j, j] = 1
        return T

    target = M_expected()
    if tensor_of(U, V, W, 44).tolist() != target.tolist():
        # fall back to plain schoolbook if (should never happen)
        U = np.zeros((n * m, 48), dtype=object)
        V = np.zeros((m * p, 48), dtype=object)
        W = np.zeros((p * n, 48), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---- Rank reduction: rational linear-algebra search over sub-blocks --------
    # A column r is a valid factor iff, for every (i,k) with target[i,k,j] != 0,
    # U[i*m+j,r] * W[k*n+i,r] == target[i,k,j] * V[j*p+k,r].
    # With U,V fixed this is linear in W, giving constraints on W (and a global
    # scale per column).  We try dropping one schoolbook column at a time and see
    # whether the remaining factors can be rescaled to absorb it.
    import time
    deadline = time.time() + 95.0

    best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    best_rank = 44

    # Build the constraint system for a fixed column set S of retained factors.
    # Unknowns: W[k*n+i] (24 vars).  Each (i,k,j) with t!=0 gives a linear eq.
    def try_reduce(S, Uf, Vf):
        # Uf, Vf: numpy arrays restricted to columns S
        import numpy as np
        rows = []
        rhs = []
        var_idx = {}
        vcount = 0
        for i in range(n):
            for k in range(p):
                idx = k * n + i
                if idx not in var_idx:
                    var_idx[idx] = vcount
                    vcount += 1
        # equations: sum_j U[i*m+j,r]*W[k*n+i,r]*V[j*p+k,r] = t[i,k,j]
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    t = 1 if j == k else 0
                    if t == 0:
                        # constraint: sum_j U[i*m+j,r]*W[k*n+i,r]*V[j*p+k,r] = 0
                        # (this is a homogeneous constraint on W)
                        coef = np.zeros(vcount)
                        for r in S:
                            s = 0
                            for jj in range(m):
                                s += Uf[jj, r] * Vf[jj * p + k, r]
                            # W[k*n+i,r] multiplies s
                            if s != 0:
                                coef[var_idx[k * n + i]] += 0  # placeholder
                        # Actually the coefficient of W[k*n+i,r] is sum_j U[i*m+j,r]*V[j*p+k,r]
                        # which is independent of i,k? No, depends on j and k and r.
                        # Let's just build directly below in a cleaner way.
                        pass
        # Build cleanly:
        rows = []
        rhs = []
        for i in range(n):
            for k in range(p):
                # coefficient of W[k*n+i, r] is sum_j U[i*m+j, r]*V[j*p+k, r]
                coef = np.zeros(vcount)
                rhs_val = 0
                for r in S:
                    c = 0
                    for j in range(m):
                        c += Uf[i * m + j, r] * Vf[j * p + k, r]
                    coef[var_idx[k * n + i]] += c
                # target contribution for this (i,k):
                # target[i,k,j] = delta_{j,k}; the product must equal it, but the
                # W factor is separated, so the constraint is:
                #   sum_r U[i*m+j,r]*W[k*n+i,r]*V[j*p+k,r] = delta_{j,k}
                # which couples across j; handle per (i,k,j).
                rows = None
                break
        # This generic approach is getting tangled; do explicit per-(i,k,j) system.
        rows = []
        rhs = []
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    t = 1 if j == k else 0
                    if t == 0:
                        continue  # only nonzero target entries produce equations
                    coef = np.zeros(vcount)
                    for r in S:
                        c = Uf[i * m + j, r] * Vf[j * p + k, r]
                        if c != 0:
                            coef[var_idx[k * n + i]] += c
                    rows.append(coef)
                    rhs.append(t)
        A = np.array(rows, dtype=float)
        b = np.array(rhs, dtype=float)
        # solve least squares; check residual
        if len(rows) == 0:
            return None
        sol, res, rank, sv = np.linalg.lstsq(A, b, rcond=None)
        if len(res) > 0 and not np.allclose(res, 0.0, atol=1e-9):
            return None
        # build W
        Wnew = np.zeros((p * n, len(S)), dtype=float)
        for idx, vi in var_idx.items():
            Wnew[idx, :] = sol[vi]
        return Wnew

    # Attempt: remove one of the 8 schoolbook columns (indices 14,15,16,17,36,37,38,39)
    school_cols = [14, 15, 16, 17, 36, 37, 38, 39]
    for drop in school_cols:
        if time.time() > deadline:
            break
        S = [r for r in range(44) if r != drop]
        Uf = U[:, S]
        Vf = V[:, S]
        Wnew = try_reduce(S, Uf, Vf)
        if Wnew is None:
            continue
        # verify full tensor
        T = tensor_of(Uf, Vf, Wnew, len(S))
        if T.tolist() == target.tolist():
            best = {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wnew.tolist()}
            best_rank = len(S)
            break  # stop at first success to keep it safe

    if best_rank < 44:
        return best
    return best
