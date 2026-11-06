"""No-isosceles set: vectorized greedy + local search with forbidden-set bookkeeping."""
import numpy as np
import time

N = 64
DEADLINE = time.time() + 290  # safety margin under 300s

# Precompute the full grid as (num_points, 2) int32 array.
_GRID = np.array([(x, y) for y in range(N) for x in range(N)], dtype=np.int32)
NP = _GRID.shape[0]  # 4096


def _sqr(a, b):
    """Squared distance between two (...,2) arrays (last dim = coords)."""
    d = a - b
    return np.sum(d * d, axis=-1)


def _valid(S):
    """Return True if index-set S contains no isosceles triangle."""
    k = len(S)
    if k < 3:
        return True
    P = _GRID[S]
    D = _sqr(P[:, None, :], P[None, :, :])  # (k,k)
    # For each anchor i, check the off-diagonal distances are all distinct.
    for i in range(k):
        row = D[i].copy()
        row[i] = -1
        if len(np.unique(row)) != k - 1:
            return False
    return True


def _forbidden_from_pair(a_idx, b_idx):
    """Indices of grid points p with ||p-a||^2 == ||p-b||^2 (the perpendicular bisector)."""
    a = _GRID[a_idx]
    b = _GRID[b_idx]
    # ||p-a||^2 - ||p-b||^2 = 0  <=>  2 p.(b-a) = |b|^2 - |a|^2
    v = b - a                       # (2,)
    rhs = (np.dot(b, b) - np.dot(a, a)) // 2
    lhs = _GRID @ v                 # (NP,)
    return np.where(lhs == rhs)[0]


def _greedy_build(rng, seed_points):
    """Greedy: start from seed, add random grid points that don't create isosceles."""
    S = list(seed_points)
    forbidden = set()
    # Build forbidden set from existing pairs.
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            forbidden |= set(_forbidden_from_pair(S[i], S[j]))
    forbidden.difference_update(S)

    pool = np.array([i for i in range(NP) if i not in forbidden], dtype=np.int32)
    while len(pool) > 0 and time.time() < DEADLINE:
        # Try a few random candidates; add the first that is safe.
        m = min(8, len(pool))
        cand = rng.choice(pool, size=m, replace=False)
        added = False
        new_pool_mask = np.ones(len(pool), dtype=bool)
        for c in cand:
            ci = int(c)
            if ci in forbidden:
                continue
            # Check: does adding ci keep S valid? i.e. is ci forbidden by any existing pair?
            # That's exactly the forbidden set. If not forbidden, adding is safe.
            # (Adding a point only creates isosceles if it lies on a bisector of an existing pair,
            #  which is captured by the forbidden set.)
            S.append(ci)
            # Update forbidden set: pairs (ci, s) for s in S[:-1]
            for s in S[:-1]:
                forbidden |= set(_forbidden_from_pair(ci, s))
            forbidden.difference_update(S)
            # Remove newly forbidden from pool
            if len(pool) > 0:
                new_pool_mask &= np.array([p not in forbidden for p in pool])
            pool = pool[new_pool_mask]
            new_pool_mask = np.ones(len(pool), dtype=bool)
            added = True
            break
        if not added:
            # none of the candidates were safe; drop them from pool to avoid retesting
            drop = set(int(c) for c in cand)
            new_pool_mask = np.array([p not in drop for p in pool])
            pool = pool[new_pool_mask]
    return S


def _local_search(S, rng, time_budget):
    """Swap-based local search: try removing one point and adding another to grow the set."""
    S = list(S)
    best = list(S)
    best_len = len(S)
    forbidden = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            forbidden |= set(_forbidden_from_pair(S[i], S[j]))
    forbidden.difference_update(S)

    end = time.time() + time_budget
    while time.time() < end:
        # Candidate additions: points not in S and not forbidden.
        in_S = set(S)
        cands = [i for i in range(NP) if i not in in_S and i not in forbidden]
        if not cands:
            break
        # Try to add a random candidate (pure growth)
        c = int(rng.choice(cands))
        # Adding c is safe iff c not in forbidden (already ensured).
        S.append(c)
        for s in S[:-1]:
            forbidden |= set(_forbidden_from_pair(c, s))
        forbidden.difference_update(S)
        if len(S) > best_len:
            best_len = len(S)
            best = list(S)

        # Occasionally try a swap to escape: remove a random point, add a random candidate.
        if len(S) > 3 and rng.random() < 0.3:
            rm = int(rng.choice(S))
            S.remove(rm)
            # recompute forbidden from scratch (cheap enough for small S)
            forbidden = set()
            for i in range(len(S)):
                for j in range(i + 1, len(S)):
                    forbidden |= set(_forbidden_from_pair(S[i], S[j]))
            forbidden.difference_update(S)
            in_S = set(S)
            cands = [i for i in range(NP) if i not in in_S and i not in forbidden]
            if cands:
                c = int(rng.choice(cands))
                S.append(c)
                for s in S[:-1]:
                    forbidden |= set(_forbidden_from_pair(c, s))
                forbidden.difference_update(S)
                if len(S) > best_len:
                    best_len = len(S)
                    best = list(S)
            else:
                # put it back
                S.append(rm)
                for s in S:
                    if s != rm:
                        forbidden |= set(_forbidden_from_pair(rm, s))
                forbidden.difference_update(S)

    return best


def construct():
    rng = np.random.default_rng(12345)
    overall_best = []

    # Several restarts with different seeds and strategies.
    restarts = 0
    while time.time() < DEADLINE:
        restarts += 1
        # Seed: a small random set that is itself valid.
        seed = []
        attempts = 0
        while len(seed) < 4 and attempts < 200:
            p = int(rng.integers(NP))
            if p not in seed and _valid(seed + [p]):
                seed.append(p)
            attempts += 1
        # If seed too small, try a structured seed: a few points on a line.
        if len(seed) < 3:
            seed = [0, N, 2 * N, 5 * N]
            if not _valid(seed):
                seed = [0, 1, 3]

        S = _greedy_build(rng, seed)
        # Local search to improve.
        S = _local_search(S, rng, time_budget=40)

        if len(S) > len(overall_best):
            overall_best = list(S)

        # Stop if we have plenty of time left but no progress across restarts.
        if len(overall_best) >= 14 and restarts >= 3:
            break

    return [tuple(_GRID[i]) for i in overall_best]
