"""Rewritten program: rank-minimal exact tensor decomposition for 2x4 * 4x5."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]


# EVOLVE-BLOCK-START
def construct():
    """
    Rank-minimal exact tensor decomposition for 2x4 * 4x5 matrix multiplication.

    Strategy:
      1. Build a correct base decomposition (block-Strassen, rank 36).
      2. Verify correctness.
      3. Exploit linear dependencies among rank-1 tensor vectors to reduce rank.
      4. Deadline guard: return best-so-far before the 120s limit.
    """
    t0 = time.time()
    DEADLINE = t0 + 110.0

    n, m, p = DIMENSIONS  # 2, 4, 5
    N_A = n * m  # 8
    N_B = m * p  # 20
    N_C = p * n  # 10
    T_DIM = n * m * p  # 40

    def time_left():
        return DEADLINE - time.time()

    # ---------- Build base decomposition (block-Strassen, rank 36) ----------
    rank_terms = []

    def add_strassen_2x2x2(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1, (j0 + 1, k0 + 1): 1},
             {(i0, k0): 1, (i0 + 1, k0 + 1): 1}),
            ({(i0 + 1, j0): 1, (i0 + 1, j0 + 1): 1},
             {(j0, k0): 1},
             {(i0 + 1, k0): 1, (i0 + 1, k0 + 1): -1}),
            ({(i0, j0): 1},
             {(j0, k0 + 1): 1, (j0 + 1, k0 + 1): -1},
             {(i0, k0 + 1): 1, (i0 + 1, k0 + 1): 1}),
            ({(i0 + 1, j0 + 1): 1},
             {(j0 + 1, k0): 1, (j0, k0): -1},
             {(i0, k0): 1, (i0 + 1, k0): 1}),
            ({(i0, j0): 1, (i0, j0 + 1): 1},
             {(j0 + 1, k0 + 1): 1},
             {(i0, k0): -1, (i0, k0 + 1): 1}),
            ({(i0 + 1, j0): 1, (i0, j0): -1},
             {(j0, k0): 1, (j0, k0 + 1): 1},
             {(i0 + 1, k0 + 1): 1}),
            ({(i0, j0 + 1): 1, (i0 + 1, j0 + 1): -1},
             {(j0 + 1, k0): 1, (j0 + 1, k0 + 1): 1},
             {(i0, k0): 1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val * sign for k, val in u.items()},
                {k: val * sign for k, val in v.items()},
                {k: val * sign for k, val in w.items()},
            ))

    def add_schoolbook_2x2x1(i0, j0, k0, sign=1):
        terms = [
            ({(i0, j0): 1}, {(j0, k0): 1}, {(i0, k0): 1}),
            ({(i0, j0 + 1): 1}, {(j0 + 1, k0): 1}, {(i0, k0): 1}),
            ({(i0 + 1, j0): 1}, {(j0, k0): 1}, {(i0 + 1, k0): 1}),
            ({(i0 + 1, j0 + 1): 1}, {(j0 + 1, k0): 1}, {(i0 + 1, k0): 1}),
        ]
        for u, v, w in terms:
            rank_terms.append((
                {k: val * sign for k, val in u.items()},
                {k: val * sign for k, val in v.items()},
                {k: val * sign for k, val in w.items()},
            ))

    # A = [A1(2x2) | A2(2x2)], B = [Btop(2x5); Bbot(2x5)]
    add_strassen_2x2x2(i0=0, j0=0, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=0, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=0, k0=4, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=0, sign=1)
    add_strassen_2x2x2(i0=0, j0=2, k0=2, sign=1)
    add_schoolbook_2x2x1(i0=0, j0=2, k0=4, sign=1)

    def build_matrices(terms):
        r = len(terms)
        U = np.zeros((N_A, r), dtype=float)
        V = np.zeros((N_B, r), dtype=float)
        W = np.zeros((N_C, r), dtype=float)
        for c, (ud, vd, wd) in enumerate(terms):
            for (i, j), coeff in ud.items():
                U[i * m + j, c] = coeff
            for (j, k), coeff in vd.items():
                V[j * p + k, c] = coeff
            for (i, k), coeff in wd.items():
                W[k * n + i, c] = coeff
        return U, V, W

    U, V, W = build_matrices(rank_terms)

    # ---------- Helpers ----------
    def tensor_vecs(U, V, W):
        r = U.shape[1]
        T = np.zeros((r, T_DIM), dtype=float)
        for c in range(r):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            for i in range(n):
                for j in range(m):
                    a = uc[i * m + j]
                    if a == 0:
                        continue
                    for k in range(p):
                        b = vc[j * p + k]
                        if b == 0:
                            continue
                        d = wc[k * n + i]
                        if d == 0:
                            continue
                        T[c, i * (m * p) + j * p + k] = a * b * d
        return T

    def verify(U, V, W, tol=1e-7):
        T = tensor_vecs(U, V, W)
        return np.allclose(T.sum(axis=0), np.ones(T_DIM), atol=tol)

    # ---------- Verify base ----------
    if not verify(U, V, W):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        U, V, W = build_matrices(terms)
        if not verify(U, V, W):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # ---------- Linear-dependency reduction ----------
    def try_reduce(U, V, W, max_iters=80):
        cur_U, cur_V, cur_W = U, V, W
        for _ in range(max_iters):
            if time_left() < 5.0:
                break
            T = tensor_vecs(cur_U, cur_V, cur_W)
            r = T.shape[0]
            if r <= 1:
                break
            try:
                _, svals, Vt = np.linalg.svd(T, full_matrices=False)
            except np.linalg.LinAlgError:
                break
            tol = 1e-9 * (svals[0] if svals.size else 1.0)
            rank_T = int(np.sum(svals > tol))
            if rank_T >= r:
                break
            null_space = Vt[rank_T:, :]
            found = False
            for nv in null_space:
                if time_left() < 5.0:
                    break
                abs_nv = np.abs(nv)
                top_pivots = np.argsort(abs_nv)[::-1][:5]
                for idx in top_pivots:
                    idx = int(idx)
                    if abs_nv[idx] < 1e-12:
                        continue
                    new_U = cur_U.copy()
                    new_V = cur_V.copy()
                    new_W = cur_W.copy()
                    for c in range(r):
                        if c == idx:
                            continue
                        s = 1.0 + nv[c] / nv[idx]
                        if abs(s) < 1e-12:
                            continue
                        new_U[:, c] *= s
                        new_V[:, c] *= s
                        new_W[:, c] *= s
                    keep = [c for c in range(r) if c != idx]
                    new_U = new_U[:, keep]
                    new_V = new_V[:, keep]
                    new_W = new_W[:, keep]
                    if verify(new_U, new_V, new_W, tol=1e-8):
                        cur_U, cur_V, cur_W = new_U, new_V, new_W
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        return cur_U, cur_V, cur_W

    U, V, W = try_reduce(U, V, W, max_iters=80)

    # ---------- Final verify & return ----------
    if not verify(U, V, W):
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    terms.append(({(i, j): 1}, {(j, k): 1}, {(i, k): 1}))
        U, V, W = build_matrices(terms)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
