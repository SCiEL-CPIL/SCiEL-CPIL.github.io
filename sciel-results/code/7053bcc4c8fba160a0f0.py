"""Construct a kissing configuration in R^11 from the D_11 lattice."""
import time
import itertools
import numpy as np

D = 11

# EVOLVE-BLOCK-START
def _build_D11():
    """Return the 500 nonzero D_11 lattice vectors (all-integer, weight 2 or 4)."""
    vecs = []
    # weight-2 vectors: +1,-1 on two coordinates
    for i, j in itertools.combinations(range(D), 2):
        for a, b in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            v = [0] * D
            v[i] = a
            v[j] = b
            vecs.append(v)
    # weight-4 vectors: four +1's on four coordinates
    for quad in itertools.combinations(range(D), 4):
        v = [0] * D
        for idx in quad:
            v[idx] = 1
        vecs.append(v)
    return np.array(vecs, dtype=np.int64)


def _min_dist(A):
    """Minimum squared pairwise distance of the rows of A."""
    n = A.shape[0]
    G = A.astype(np.int64) @ A.astype(np.int64).T
    d2 = G + G.T - 2 * G
    np.fill_diagonal(d2, 0)
    return int(d2.min())


def construct():
    deadline = time.time() + 110.0
    vecs = _build_D11()

    # Verify the constraint: min pairwise distance >= max vector norm.
    min_d2 = _min_dist(vecs)
    max_n2 = int(np.max(np.sum(vecs * vecs, axis=1)))
    base_ok = min_d2 >= max_n2

    # Greedily try to add integer vectors with squared norm <= max_n2
    # that keep the minimum pairwise distance at least max_n2.
    if base_ok:
        max_n2 = int(np.max(np.sum(vecs * vecs, axis=1)))
        best = vecs
        # Candidate pool: integer vectors with small norm (<= max_n2).
        # Limit coordinate values to keep the search fast.
        bound = int(np.sqrt(max_n2)) + 1
        pool = []
        for coords in itertools.product(range(-bound, bound + 1), repeat=D):
            n2 = sum(c * c for c in coords)
            if 0 < n2 <= max_n2:
                pool.append(coords)
        # Sort pool by norm (small first) for better packing; cap size for speed.
        pool.sort(key=lambda c: sum(x * x for x in c))
        pool = pool[:400000]

        cur = vecs
        cur_min_d2 = min_d2
        # Precompute current norms for distance checks.
        cur_G = cur.astype(np.int64) @ cur.astype(np.int64).T
        cur_norm2 = np.diag(cur_G).copy()

        for cand in pool:
            if time.time() > deadline:
                break
            c = np.array(cand, dtype=np.int64)
            cn2 = int(np.sum(c * c))
            if cn2 > max_n2:
                continue
            # distance to every existing vector
            d2 = cur_norm2 + cn2 - 2 * (cur @ c)
            if int(d2.min()) < max_n2:
                continue
            # accept
            cur = np.vstack([cur, c[None, :]])
            cur_G = np.vstack([np.hstack([cur_G, (cur @ c)[None, :]]),
                               np.hstack([(cur @ c)[:, None], np.array([[cn2]])])])
            cur_norm2 = np.diag(cur_G).copy()
            cur_min_d2 = min(cur_min_d2, int(d2.min()))

        if cur.shape[0] > best.shape[0]:
            best = cur

    # Final safety check before returning.
    if best.shape[0] > 0:
        md2 = _min_dist(best)
        mn2 = int(np.max(np.sum(best * best, axis=1)))
        if md2 < mn2:
            # Fall back to the verified D_11 core.
            best = vecs
    return best
# EVOLVE-BLOCK-END
