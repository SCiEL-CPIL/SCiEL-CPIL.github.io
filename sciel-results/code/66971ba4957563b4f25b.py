"""Direct approach for minimum maximal no-isosceles set."""
N = 16
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
TOTAL = N * N

def _point_idx(p):
    return p[0] * N + p[1]

def _idx_point(i):
    return (i // N, i % N)

def _precompute():
    """Precompute iso_mask[i][j] as a Python int bitmask of points c where {i,j,c} is isosceles."""
    xs = np.arange(TOTAL) // N
    ys = np.arange(TOTAL) % N
    # Pairwise squared distances: shape (TOTAL, TOTAL)
    dx = xs[:, None].astype(np.int32) - xs[None, :].astype(np.int32)
    dy = ys[:, None].astype(np.int32) - ys[None, :].astype(np.int32)
    dist2 = dx * dx + dy * dy  # (TOTAL, TOTAL)
    
    # For each pair (i,j), find all c where dist(i,c)==dist(j,c) or dist(i,c)==dist(i,j) or dist(j,c)==dist(i,j)
    # iso_mask[i][j] = bitmask of such c (excluding i and j)
    iso_mask = np.zeros((TOTAL, TOTAL), dtype=object)
    
    for i in range(TOTAL):
        for j in range(TOTAL):
            if i == j:
                iso_mask[i][j] = 0
                continue
            dij = dist2[i, j]
            # c such that dist(i,c) == dij  (c equidistant from i at distance dij)
            mask1 = (dist2[i, :] == dij)
            # c such that dist(j,c) == dij
            mask2 = (dist2[j, :] == dij)
            # c such that dist(i,c) == dist(j,c)
            mask3 = (dist2[i, :] == dist2[j, :])
            combined = mask1 | mask2 | mask3
            combined[i] = False
            combined[j] = False
            # Convert to bitmask
            bits = 0
            idxs = np.nonzero(combined)[0]
            for k in idxs:
                bits |= (1 << k)
            iso_mask[i][j] = bits
    
    return iso_mask

_iso_mask = _precompute()

def _greedy(order):
    """Greedy: add points in order if not blocked. Returns (chosen_list, blocked_mask)."""
    chosen = []
    blocked = 0
    for p in order:
        if (blocked >> p) & 1:
            continue
        new_block = 0
        for q in chosen:
            new_block |= _iso_mask[p][q]
        blocked |= new_block
        blocked |= (1 << p)
        chosen.append(p)
    return chosen, blocked

def _is_maximal(chosen, blocked):
    """Check if no unchosen point is unblocked."""
    cmask = 0
    for c in chosen:
        cmask |= (1 << c)
    all_mask = (1 << TOTAL) - 1
    unchosen = all_mask & ~cmask
    return (blocked & unchosen) == unchosen

def _compute_required(chosen):
    """For each point p in chosen, compute how many OTHER chosen points require p for maximality.
    Actually, we need: for each unchosen point u, how many pairs (a,b) in chosen have u in iso_mask[a][b].
    But for removal, we need: if we remove point r, is the set still maximal?
    A point r is 'removable' if for every unchosen point u (not r), there exists a pair (a,b) in chosen\{r}
    such that u is in iso_mask[a][b].
    
    More efficiently: for each point r in chosen, count for each unchosen u how many pairs (a,b) in chosen
    cover u. If removing r still leaves every u covered, r is removable.
    
    Let's use: cover_count[u] = number of pairs (a,b) in chosen such that u in iso_mask[a][b].
    If we remove r, the new cover_count for u is cover_count[u] - (number of pairs involving r that cover u).
    r is removable iff for all unchosen u: new_cover_count[u] > 0.
    """
    n = len(chosen)
    if n < 2:
        return chosen  # can't remove if < 2 points (and maximality likely fails)
    
    all_mask = (1 << TOTAL) - 1
    cmask = 0
    for c in chosen:
        cmask |= (1 << c)
    unchosen_mask = all_mask & ~cmask
    
    # For each point r in chosen, compute which unchosen points are ONLY covered by pairs involving r
    # i.e., points u where the only pairs (a,b) covering u include r.
    # This is expensive. Alternative: just try removal and check.
    
    # Faster approach: for each r, compute the "exclusive coverage" of r.
    # A point u is exclusively covered by r if every pair (a,b) in chosen that covers u includes r.
    # Equivalently: u is NOT covered by any pair (a,b) where a,b != r.
    
    # Let's precompute: for each pair (a,b), the set of unchosen points they cover.
    # Then for removal of r: a point u is still covered if there exists (a,b) with a!=r, b!=r covering u.
    
    # For efficiency, compute for each r the set of unchosen points that are NOT covered by any pair not involving r.
    # If that set is empty, r is removable.
    
    # Precompute pair coverage
    pair_cover = {}  # (a,b) -> bitmask of covered unchosen points
    for idx_a in range(n):
        for idx_b in range(idx_a + 1, n):
            a = chosen[idx_a]
            b = chosen[idx_b]
            cover = _iso_mask[a][b] & unchosen_mask
            pair_cover[(a, b)] = cover
    
    # For each r, compute union of all pair covers NOT involving r
    for r in chosen:
        # Union of covers of all pairs not involving r
        union_cover = 0
        for (a, b), cover in pair_cover.items():
            if a != r and b != r:
                union_cover |= cover
        # Points in unchosen_mask not in union_cover are only covered by pairs involving r
        only_r = unchosen_mask & ~union_cover
        if only_r == 0:
            # r is removable
            return r
    
    return None

def _try_swap_remove(chosen, random_order):
    """Try removing one point and optionally adding another to maintain maximality with fewer points.
    Actually, just try removal first. If that doesn't work, try swap: remove r, add s where s was unchosen,
    resulting set has same size but might enable further removals.
    For now, just do removal."""
    r = _compute_required(chosen)
    if r is not None:
        return [c for c in chosen if c != r]
    return None

def construct():
    all_idx = list(range(TOTAL))
    best = None
    best_size = TOTAL + 1
    start = time.time()
    deadline = start + 310  # safety margin before 330s kill

    random.seed(42)
    
    # Try several deterministic orderings first
    orderings = [
        all_idx[:],  # row-major
        sorted(all_idx, key=lambda i: (i // N + i % N, i)),  # diagonal
        sorted(all_idx, key=lambda i: (abs(i // N - 7.5) + abs(i % N - 7.5), i)),  # center-out
        sorted(all_idx, key=lambda i: (abs(i // N - 7.5) * abs(i % N - 7.5), i)),  # corner-biased
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        chosen, blocked = _greedy(order)
        if _is_maximal(chosen, blocked):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen[:]
    
    # Random restarts
    for attempt in range(500000):
        if time.time() > deadline - 5:
            break
        order = all_idx[:]
        random.shuffle(order)
        chosen, blocked = _greedy(order)
        if _is_maximal(chosen, blocked):
            if len(chosen) < best_size:
                best_size = len(chosen)
                best = chosen[:]
                if best_size <= 12:
                    break
    
    if best is None:
        best = all_idx[:]
    
    # Local search: iterated removal
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            result = _try_swap_remove(best, None)
            if result is not None:
                best = result
                best_size = len(best)
                improved = True
    
    # Additional local search: try removal from multiple random maximal sets
    if time.time() < deadline - 15:
        for attempt in range(100000):
            if time.time() > deadline - 10:
                break
            order = all_idx[:]
            random.shuffle(order)
            chosen, blocked = _greedy(order)
            if _is_maximal(chosen, blocked):
                # Try to shrink this one
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    result = _try_swap_remove(chosen, None)
                    if result is not None:
                        chosen = result
                        improved = True
                if len(chosen) < best_size:
                    best_size = len(chosen)
                    best = chosen[:]
    
    return [_idx_point(i) for i in best]

# EVOLVE-BLOCK-END
