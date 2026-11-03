"""Kissing number in dimension 11: construct nonzero integer vectors with
min pairwise distance >= max norm. Score = k (number of vectors)."""
import time
import itertools
import numpy as np

D = 11
START = time.time()
DEADLINE = 150.0


def _shell(max_coeff):
    """All integer vectors in Z^11 with max |coord| <= max_coeff, nonzero,
    up to global sign (canonical: first nonzero coord > 0)."""
    coords = np.arange(-max_coeff, max_coeff + 1, dtype=np.int64)
    arr = np.stack(np.meshgrid(*([coords] * D), indexing='ij'), axis=-1).reshape(-1, D)
    mask = np.any(arr != 0, axis=1)
    arr = arr[mask]
    first = np.argmax(arr != 0, axis=1)
    rows = np.arange(arr.shape[0])
    sign = arr[rows, first]
    arr[sign < 0] *= -1
    arr = np.unique(arr, axis=0)
    return arr


def _generate_candidates(max_coeff):
    """Build candidate pool. Shell m=2 is the full kissing set (220).
    Adding shell m=4 gives more candidates; greedy keeps only a valid subset."""
    pool = []
    for m in (2, 3, 4):
        if time.time() > DEADLINE - 30:
            break
        pool.append(_shell(m))
    cands = np.vstack(pool)
    # Deduplicate across shells
    cands = np.unique(cands, axis=0)
    return cands


def construct():
    start = time.time()
    # Baseline: full m=2 kissing set is always valid.
    base = _shell(2)
    best = base.copy()

    # Try to improve by greedy on a larger pool.
    try:
        cands = _generate_candidates(4)
    except Exception:
        cands = base

    # Greedy: pick vectors maximizing min-distance to already chosen.
    # Constraint: for all pairs, |u-v|^2 >= max(|u|^2, |v|^2).
    # Equivalent: |u-v|^2 >= max_norm_sq. We track max_norm_sq of chosen.
    # We'll use a simple greedy with vectorized distance updates.

    chosen = []
    max_norm_sq = 0
    n_cands = cands.shape[0]
    # Precompute norms
    norms_sq = np.einsum('ij,ij->i', cands, cands)

    # Greedy selection: at each step pick candidate that maximizes
    # min over chosen of (|c - v|^2), subject to that min >= max(max_norm_sq, |c|^2).
    # To keep it fast, we do a few passes.

    # Start with the m=2 set as a strong seed, then try to add more.
    # Actually, do a fresh greedy but it's expensive. Instead, start from base
    # and try to add candidates from the larger pool.

    chosen_list = [tuple(r) for r in base]
    chosen_arr = base.copy()
    chosen_norms = norms_sq[np.argsort(np.argsort(norms_sq))]  # not needed
    # Recompute chosen norms from base
    chosen_norms = np.einsum('ij,ij->i', chosen_arr, chosen_arr)
    max_norm_sq = int(np.max(chosen_norms))

    # Candidate pool excluding those already in base
    base_set = set(tuple(r) for r in base)
    extra_mask = np.array([tuple(r) not in base_set for r in cands], dtype=bool)
    extra = cands[extra_mask]
    extra_norms = np.einsum('ij,ij->i', extra, extra)

    # Sort extra by norm ascending to add small ones first (easier to satisfy)
    order = np.argsort(extra_norms)
    extra = extra[order]
    extra_norms = extra_norms[order]

    added = 0
    for idx in range(extra.shape[0]):
        if time.time() > DEADLINE - 20:
            break
        c = extra[idx]
        cn = int(extra_norms[idx])
        # Need |c - v|^2 >= max(max_norm_sq, cn) for all chosen v
        thr = max(max_norm_sq, cn)
        # |c-v|^2 = |c|^2 + |v|^2 - 2 c.v
        # Need |c|^2 + |v|^2 - 2 c.v >= thr for all v
        # => 2 c.v <= |c|^2 + |v|^2 - thr
        # We need min over v of (|c|^2+|v|^2-2c.v) >= thr
        # Compute min distance squared to chosen
        # |c-v|^2 = cn + chosen_norms - 2 * (c @ chosen^T)
        dots = chosen_arr @ c
        dists_sq = cn + chosen_norms - 2 * dots
        if np.min(dists_sq) >= thr:
            chosen_arr = np.vstack([chosen_arr, c])
            chosen_norms = np.append(chosen_norms, cn)
            if cn > max_norm_sq:
                max_norm_sq = cn
            added += 1

    result = chosen_arr
    if result.shape[0] <= best.shape[0]:
        result = best
    return result.astype(np.int64)
