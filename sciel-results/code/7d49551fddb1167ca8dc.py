import time
import numpy as np

# Precompute K4 edge sets and vertex-edge incidence once.
def _build_tables():
    idx = {}
    e = 0
    for i in range(16):
        for j in range(i + 1, 16):
            idx[(i, j)] = e
            e += 1
    assert e == 120
    k4s = []
    for a in range(16):
        for b in range(a + 1, 16):
            for c in range(b + 1, 16):
                for d in range(c + 1, 16):
                    k4s.append((a, b, c, d))
    k4s = np.array(k4s, dtype=np.int32)
    K = len(k4s)
    E = 120
    M = np.zeros((K, 16, 6), dtype=np.int32)
    for r, (a, b, c, d) in enumerate(k4s):
        M[r, a] = idx[(a, b)]
        M[r, b] = idx[(a, c)]
        M[r, c] = idx[(a, d)]
        M[r, d] = idx[(b, c)]
        M[r, a + 1] = idx[(b, d)]
        M[r, b + 1] = idx[(c, d)]
    return M, E

_M, _E = _build_tables()
_K = _M.shape[0]
_MF = _M.astype(np.int64)


def _bad(c):
    # c: (120,) int array of colours. Returns number of K4s with <5 distinct colours.
    onehot = (c[None, None, :] == _MF).astype(np.float32)
    distinct = (onehot.max(axis=2) > 0).sum(axis=1)
    return int(np.maximum(5 - distinct, 0).sum())


def _compact(c):
    # Relabel colours to 0..(used-1) preserving order of first appearance.
    used = np.unique(c)
    remap = {int(col): k for k, col in enumerate(used)}
    return np.array([remap[int(x)] for x in c], dtype=np.int64)


def construct():
    start = time.time()
    deadline = start + 140.0
    rng = np.random.default_rng(20240611)

    # Seed: the known valid 17-colour scheme.
    base = np.array([(i + j) % 17 for i in range(16) for j in range(i + 1, 16)], dtype=np.int64)
    base = _compact(base)
    best = base.copy()
    best_bad = _bad(best)
    best_colors = int(np.max(best)) + 1
    assert best_bad == 0

    # Single-edge recoloring moves.
    for _ in range(2000000):
        if time.time() > deadline:
            break
        e = int(rng.integers(_E))
        old = int(best[e])
        new = int(rng.integers(30))
        if new == old:
            continue
        cand = best.copy()
        cand[e] = new
        b = _bad(cand)
        if b < best_bad or (b == best_bad and rng.random() < 0.05):
            best = cand
            best_bad = b
            if best_bad == 0:
                cc = _compact(best)
                if int(np.max(cc)) + 1 < best_colors:
                    best = cc
                    best_colors = int(np.max(cc)) + 1

    # Pairwise colour-swap moves.
    for _ in range(1000000):
        if time.time() > deadline:
            break
        e1 = int(rng.integers(_E))
        e2 = int(rng.integers(_E))
        if e1 == e2:
            continue
        cand = best.copy()
        cand[e1], cand[e2] = best[e2], best[e1]
        b = _bad(cand)
        if b < best_bad or (b == best_bad and rng.random() < 0.05):
            best = cand
            best_bad = b
            if best_bad == 0:
                cc = _compact(best)
                if int(np.max(cc)) + 1 < best_colors:
                    best = cc
                    best_colors = int(np.max(cc)) + 1

    # Final compaction.
    best = _compact(best)
    best_colors = int(np.max(best)) + 1

    # Safety: if somehow invalid, fall back to the seed (guaranteed valid).
    if _bad(best) != 0:
        best = base
        best_colors = int(np.max(base)) + 1

    return [int(x) for x in best]
