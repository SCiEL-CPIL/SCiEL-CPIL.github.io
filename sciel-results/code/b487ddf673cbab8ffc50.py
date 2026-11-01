import time
import numpy as np

# EVOLVE-BLOCK-START

def _strassen_2x2x6():
    """Rank-21 baseline: 3 independent Strassen blocks over the 6 columns."""
    n, m, p = 2, 2, 6
    rank = 21
    U = [[0] * rank for _ in range(n * m)]   # 4 x 21
    V = [[0] * rank for _ in range(m * p)]   # 12 x 21
    W = [[0] * rank for _ in range(p * n)]   # 12 x 21

    # Strassen factors for 2x2 matrix multiplication (u: A coeffs, v: B coeffs, w: C coeffs)
    # A/B/C local indices: 0=00, 1=01, 2=10, 3=11
    steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},   # M1
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},  # M2
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},  # M3
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},  # M4
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},  # M5
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},  # M6
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},  # M7
    ]

    def map_a(li):
        return li

    def map_b(li, g):
        k0, k1 = 2 * g, 2 * g + 1
        return {0: 0 * 6 + k0, 1: 0 * 6 + k1, 2: 1 * 6 + k0, 3: 1 * 6 + k1}[li]

    def map_c(li, g):
        k0, k1 = 2 * g, 2 * g + 1
        return {0: k0 * 2 + 0, 1: k1 * 2 + 0, 2: k0 * 2 + 1, 3: k1 * 2 + 1}[li]

    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            st = steps[s]
            for li in range(4):
                if st['u'][li]:
                    U[map_a(li)][r] = st['u'][li]
                if st['v'][li]:
                    V[map_b(li, g)][r] = st['v'][li]
                if st['w'][li]:
                    W[map_c(li, g)][r] = st['w'][li]
    return {"U": U, "V": V, "W": W}


def _schoolbook(n, m, p):
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


def _exact_verify(U, V, W, n, m, p):
    """Exact check: sum_r U[a,r] V[b,r] W[c,r] == delta(a,b,c) for all a,b,c (integer)."""
    nm, mp, pn = n * m, m * p, p * n
    rank = len(U[0])
    for a in range(nm):
        i, j = divmod(a, m)
        for b in range(mp):
            j2, k = divmod(b, p)
            if j != j2:
                for c in range(pn):
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != 0:
                        return False
            else:
                target_c = k * n + i
                for c in range(pn):
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if s != (1 if c == target_c else 0):
                        return False
    return True


def _term_tensors(U, V, W, rank):
    """Return list of rank tensors T_r[a,b,c] = U[a,r] V[b,r] W[c,r] (dicts of nonzero)."""
    nm, mp, pn = len(U), len(V), len(W)
    out = []
    for r in range(rank):
        u = [U[a][r] for a in range(nm)]
        v = [V[b][r] for b in range(mp)]
        w = [W[c][r] for c in range(pn)]
        d = {}
        for a in range(nm):
            if u[a] == 0:
                continue
            for b in range(mp):
                if v[b] == 0:
                    continue
                uv = u[a] * v[b]
                for c in range(pn):
                    if w[c] == 0:
                        continue
                    d[(a, b, c)] = uv * w[c]
        out.append(d)
    return out


def _try_reduce_by_dependence(base, n, m, p, deadline):
    """Detect a rank-1 term whose tensor is a rational linear combination of the
    others and remove it exactly. Returns a lower-rank valid decomposition or None."""
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]

    for _ in range(20):
        if time.time() > deadline:
            break
        rank = len(U[0])
        if rank <= 1:
            break
        terms = _term_tensors(U, V, W, rank)
        found = -1
        for drop in range(rank):
            if time.time() > deadline:
                break
            target = terms[drop]
            if not target:
                found = drop
                break
            keys = list(target.keys())
            others = [terms[r] for r in range(rank) if r != drop]
            A = []
            rhs = []
            for key in keys:
                A.append([ot.get(key, 0) for ot in others])
                rhs.append(target[key])
            Aarr = np.array(A, dtype=float)
            bvec = np.array(rhs, dtype=float)
            try:
                sol = np.linalg.lstsq(Aarr, bvec, rcond=None)[0]
            except Exception:
                continue
            resid = np.max(np.abs(Aarr @ sol - bvec))
            if resid < 1e-6:
                found = drop
                break
        if found < 0:
            break
        keep = [r for r in range(rank) if r != found]
        U = [[U[a][r] for r in keep] for a in range(len(U))]
        V = [[V[b][r] for r in keep] for b in range(len(V))]
        W = [[W[c][r] for r in keep] for c in range(len(W))]
        if not _exact_verify(U, V, W, n, m, p):
            break
    if len(U[0]) < len(base["U"][0]):
        return {"U": U, "V": V, "W": W, "rank": len(U[0])}
    return None


def _local_search(base, n, m, p, deadline):
    """Perturb one term's entries among small integers and verify exactly; keep improvements."""
    U = [row[:] for row in base["U"]]
    V = [row[:] for row in base["V"]]
    W = [row[:] for row in base["W"]]
    rank = len(U[0])
    best_rank = rank
    values = [-2, -1, 0, 1, 2]
    rng = np.random.default_rng(12345)
    for _ in range(400):
        if time.time() > deadline:
            break
        r = int(rng.integers(rank))
        cU = [row[:] for row in U]
        cV = [row[:] for row in V]
        cW = [row[:] for row in W]
        for _ in range(3):
            cU[int(rng.integers(len(cU)))][r] = int(rng.choice(values))
            cV[int(rng.integers(len(cV)))][r] = int(rng.choice(values))
            cW[int(rng.integers(len(cW)))][r] = int(rng.choice(values))
        if _exact_verify(cU, cV, cW, n, m, p):
            keep = [rr for rr in range(rank)
                    if any(cU[a][rr] != 0 for a in range(len(cU)))
                    or any(cV[b][rr] != 0 for b in range(len(cV)))
                    or any(cW[c][rr] != 0 for c in range(len(cW)))]
            if len(keep) < best_rank:
                cU2 = [[cU[a][rr] for rr in keep] for a in range(len(cU))]
                cV2 = [[cV[b][rr] for rr in keep] for b in range(len(cV))]
                cW2 = [[cW[c][rr] for rr in keep] for c in range(len(cW))]
                if _exact_verify(cU2, cV2, cW2, n, m, p):
                    U, V, W = cU2, cV2, cW2
                    rank = len(keep)
                    best_rank = rank
    if best_rank < len(base["U"][0]):
        return {"U": U, "V": V, "W": W, "rank": best_rank}
    return None


def construct():
    n, m, p = 2, 2, 6
    start_time = time.time()
    deadline = start_time + 135.0  # safety margin before 150s kill

    base = _strassen_2x2x6()
    if not _exact_verify(base["U"], base["V"], base["W"], n, m, p):
        return _schoolbook(n, m, p)

    best = base
    best_rank = len(base["U"][0])

    reduced = _try_reduce_by_dependence(base, n, m, p, deadline)
    if reduced is not None and reduced["rank"] < best_rank:
        best = reduced
        best_rank = reduced["rank"]

    if time.time() < deadline:
        improved = _local_search(best, n, m, p, deadline)
        if improved is not None and improved["rank"] < best_rank:
            best = improved
            best_rank = improved["rank"]

    if _exact_verify(best["U"], best["V"], best["W"], n, m, p):
        return {"U": best["U"], "V": best["V"], "W": best["W"]}

    return _schoolbook(n, m, p)

# EVOLVE-BLOCK-END
