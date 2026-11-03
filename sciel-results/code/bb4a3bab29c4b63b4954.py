"""Tensor decomposition for 3x5x7 matrix multiplication (rank-minimization)."""
import time

DIMENSIONS = [3, 5, 7]
_START_TIME = time.time()
_DEADLINE = 150.0 - 10.0  # safety margin


def _subsets2(lst):
    n = len(lst)
    for a in range(n):
        for b in range(a + 1, n):
            yield (lst[a], lst[b])


def _strassen_terms(I2, J2, K2):
    """Return 7 rank-1 terms (u, v, w) as dicts for the 2x2x2 core."""
    i0, i1 = I2
    j0, j1 = J2
    k0, k1 = K2

    def Aloc(r, c):  # local (r,c) -> global A index i*m+j
        i = i0 if r == 0 else i1
        j = j0 if c == 0 else j1
        return (i, j)

    def Bloc(r, c):
        j = j0 if r == 0 else j1
        k = k0 if c == 0 else k1
        return (j, k)

    def Cloc(r, c):
        i = i0 if r == 0 else i1
        k = k0 if c == 0 else k1
        return (i, k)

    # local position order: (0,0)->0,(0,1)->1,(1,0)->2,(1,1)->3
    def mkA(vec):
        d = {}
        for idx, val in enumerate(vec):
            if val:
                r, c = divmod(idx, 2)
                d[Aloc(r, c)] = val
        return d

    def mkB(vec):
        d = {}
        for idx, val in enumerate(vec):
            if val:
                r, c = divmod(idx, 2)
                d[Bloc(r, c)] = val
        return d

    def mkC(vec):
        d = {}
        for idx, val in enumerate(vec):
            if val:
                r, c = divmod(idx, 2)
                d[Cloc(r, c)] = val
        return d

    terms = []
    # M1
    terms.append((mkA([1, 0, 0, 1]), mkB([1, 0, 0, 1]), mkC([1, 0, 0, 1])))
    # M2
    terms.append((mkA([0, 0, 1, 1]), mkB([1, 0, 0, 0]), mkC([0, 0, 1, -1])))
    # M3
    terms.append((mkA([1, 0, 0, 0]), mkB([0, -1, 0, 1]), mkC([0, 1, 0, 1])))
    # M4
    terms.append((mkA([0, 0, 0, 1]), mkB([0, 0, -1, 1]), mkC([1, 0, 0, 1])))
    # M5
    terms.append((mkA([1, 1, 0, 0]), mkB([0, 0, 0, 1]), mkC([0, 0, 1, 0])))
    # M6
    terms.append((mkA([0, 1, 0, -1]), mkB([0, 0, 1, 1]), mkC([-1, 0, 0, 0])))
    # M7
    terms.append((mkA([1, 0, -1, 0]), mkB([1, 1, 0, 0]), mkC([-1, 1, 0, 0])))
    return terms


def _conflict(c1, c2):
    I1, J1, K1 = c1
    I2, J2, K2 = c2
    return (set(I1) & set(I2)) and (set(J1) & set(J2)) and (set(K1) & set(K2))


def construct():
    n, m, p = DIMENSIONS
    # Target tensor entries
    target = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target[((i, j), (j, k), (i, k))] = 1

    # Candidate cores
    cores = []
    for I2 in _subsets2(range(n)):
        for J2 in _subsets2(range(m)):
            for K2 in _subsets2(range(p)):
                cores.append((I2, J2, K2))

    # Greedy set-packing: order cores so we maximize count; use a heuristic order
    # Prefer cores that are "spread out" to reduce future conflicts.
    chosen = []
    chosen_set = set()
    # deterministic shuffle-free ordering but prioritize larger spread
    for core in cores:
        if time.time() > _DEADLINE:
            break
        conflict = False
        for c2 in chosen:
            if _conflict(core, c2):
                conflict = True
                break
        if not conflict:
            chosen.append(core)
            chosen_set.add(core)

    # Build terms
    terms = []
    covered = set()
    for core in chosen:
        I2, J2, K2 = core
        for (u, v, w) in _strassen_terms(I2, J2, K2):
            terms.append((u, v, w))
        # mark covered entries
        for i in I2:
            for j in J2:
                for k in K2:
                    covered.add(((i, j), (j, k), (i, k)))

    # schoolbook for remaining
    for key, val in target.items():
        if key not in covered:
            (ia, ja), (jb, kb), (ic, kc) = key
            u = {(ia, ja): val}
            v = {(jb, kb): val}
            w = {(ic, kc): val}
            terms.append((u, v, w))

    # Verification
    recon = {}
    for (u, v, w) in terms:
        for (ia, ja), ua in u.items():
            for (jb, kb), vb in v.items():
                for (ic, kc), wc in w.items():
                    key = ((ia, ja), (jb, kb), (ic, kc))
                    recon[key] = recon.get(key, 0) + ua * vb * wc

    ok = (recon == target)

    if not ok:
        # Fallback: pure schoolbook (always valid)
        terms = []
        for key, val in target.items():
            (ia, ja), (jb, kb), (ic, kc) = key
            terms.append((
                {(ia, ja): val},
                {(jb, kb): val},
                {(ic, kc): val},
            ))

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    for r, (u, v, w) in enumerate(terms):
        for (i, j), val in u.items():
            U[i * m + j][r] = val
        for (j, k), val in v.items():
            V[j * p + k][r] = val
        for (i, k), val in w.items():
            W[k * n + i][r] = val

    return {"U": U, "V": V, "W": W}
