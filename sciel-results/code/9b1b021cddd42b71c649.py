"""Initial program for the suite task; SCiEL evolves the EVOLVE-BBLOCK."""
D = 11

# EVOLVE-BLOCK-START
import itertools
import numpy as np
import time


def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    # Baseline: 220 vectors of form (±1, ±1, 0, ..., 0), all norm sqrt(2)
    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)

    arr = np.array(vectors, dtype=np.int64)

    # Current state
    norms_sq = np.sum(arr ** 2, axis=1)
    max_norm = np.sqrt(np.max(norms_sq))  # sqrt(2)

    def can_add(candidate, arr, max_norm):
        """Return (ok, new_max_norm). Constraint: min pairwise dist >= max norm."""
        cand_arr = np.array(candidate)
        cand_norm = np.sqrt(float(np.sum(cand_arr ** 2)))
        new_max_norm = max(max_norm, cand_norm)
        # Squared distances from candidate to all existing
        diffs = arr - cand_arr
        dists_sq = np.sum(diffs ** 2, axis=1)
        min_dist = np.sqrt(np.min(dists_sq))
        if min_dist + 1e-9 < new_max_norm:
            return False, new_max_norm
        return True, new_max_norm

    # Generate candidate vectors of various types (small norms preferred)
    candidates = []
    seen = set(map(tuple, arr.tolist()))

    # Type A: (±2, 0, ..., 0)  norm 2
    for i in range(D):
        for s in (1, -1):
            v = [0] * D
            v[i] = s * 2
            t = tuple(v)
            if t not in seen:
                candidates.append(v)
                seen.add(t)

    # Type B: (±1,±1,±1,±1,0^7)  norm 2
    for pos in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            t = tuple(v)
            if t not in seen:
                candidates.append(v)
                seen.add(t)

    # Type C: (±1,±1,±1,±1,±1,0^6)  norm sqrt(5)
    for pos in itertools.combinations(range(D), 5):
        for signs in itertools.product([1, -1], repeat=5):
            v = [0] * D
            for p, s in zip(pos, signs):
                v[p] = s
            t = tuple(v)
            if t not in seen:
                candidates.append(v)
                seen.add(t)

    # Sort by norm ascending (keep max_norm as small as possible)
    candidates.sort(key=lambda v: sum(x * x for x in v))

    # Greedy add
    for cand in candidates:
        if time.time() > deadline:
            break
        ok, new_max = can_add(cand, arr, max_norm)
        if ok:
            arr = np.vstack([arr, np.array(cand, dtype=np.int64)])
            max_norm = new_max

    return arr
# EVOLVE-BLOCK-END
