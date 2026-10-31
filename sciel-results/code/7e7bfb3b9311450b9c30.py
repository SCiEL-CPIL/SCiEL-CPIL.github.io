"""Rewritten program: kissing number in dimension 11 via Leech-lattice-inspired construction."""
import time
import numpy as np

D = 11
DEADLINE = 115.0  # safety margin before 120s


def _norm2(v):
    return int(np.dot(v, v))


def construct():
    t0 = time.time()

    # --- Step 1: Generate a large pool of candidate integer vectors in R^11 ---
    # We use vectors with small integer coordinates. The key idea from the Leech lattice
    # decomposition E8 ⊕ A2 ⊕ A1 suggests combining structures from lower-dimensional lattices.
    # However, a practical approach for dimension 11 is to generate vectors from:
    #   (a) weight-2 vectors (±1, ±1 in two coords, 0 elsewhere)
    #   (b) weight-3 vectors with entries from {±1, ±2}
    #   (c) weight-4 vectors with entries from {±1, ±2, ±3}
    # and then select a maximal subset satisfying the distance constraint.
    #
    # The constraint: min pairwise distance >= max norm among selected vectors.
    # If we normalize to norm 2, the constraint becomes: for all pairs, ||vi - vj|| >= ||vi|| (after scaling).
    # In raw integer terms: for selected set S with max_norm2 = max_{v in S} ||v||^2,
    # we need ||vi - vj||^2 >= max_norm2 for all i != j in S.

    candidates = []

    # (a) Weight-2 vectors: two non-zero entries, each ±1
    # Number: C(11,2) * 4 = 55 * 4 = 220
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    candidates.append(v)

    # (b) Weight-3 vectors: three non-zero entries from {±1, ±2}, with at least one ±2 to get norm^2 >= 6
    # Actually let's include all combos of {±1, ±2} in 3 positions.
    # To keep pool manageable, use entries from {-2,-1,1,2}
    vals3 = [-2, -1, 1, 2]
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for a in vals3:
                    for b in vals3:
                        for c in vals3:
                            v = [0] * D
                            v[i], v[j], v[k] = a, b, c
                            candidates.append(v)

    # (c) Weight-4 vectors: four non-zero entries from {±1, ±2}
    # This could be large: C(11,4)*4^4 = 330*256 = 84480. Acceptable.
    vals4 = [-2, -1, 1, 2]
    for i in range(D):
        for j in range(i + 1, D):
            for k in range(j + 1, D):
                for l in range(k + 1, D):
                    for a in vals4:
                        for b in vals4:
                            for c in vals4:
                                for d in vals4:
                                    v = [0] * D
                                    v[i], v[j], v[k], v[l] = a, b, c, d
                                    candidates.append(v)

    # Convert to numpy array
    cand_arr = np.array(candidates, dtype=np.int64)
    n = len(cand_arr)

    # Precompute squared norms
    norms2 = np.sum(cand_arr * cand_arr, axis=1)

    # Sort candidates by norm2 descending (we want to consider higher-norm vectors first
    # but the constraint is min distance >= max norm, so if we include a high-norm vector,
    # all others must be at least that far. Better strategy: try to find a large set where
    # the max norm is as small as possible relative to distances, OR just greedily pick.)
    #
    # Greedy approach: sort by norm2 ascending. Pick vectors one by one; when picking a new vector,
    # the constraint is that its distance to all already-picked vectors must be >= current max norm2
    # (which will update if this vector has larger norm).
    #
    # Actually, a cleaner greedy: sort by norm2 ascending. Maintain selected set and its max_norm2.
    # For each candidate (in order), check if min distance to selected set >= max(current max_norm2, this vector's norm2).
    # If yes, add it and update max_norm2.

    order = np.argsort(norms2, kind='stable')
    selected_idx = []
    selected_vecs = []
    current_max_norm2 = 0

    for idx in order:
        if time.time() - t0 > DEADLINE:
            break
        v = cand_arr[idx]
        n2 = norms2[idx]
        new_max_norm2 = max(current_max_norm2, int(n2))

        if not selected_vecs:
            selected_idx.append(idx)
            selected_vecs.append(v)
            current_max_norm2 = int(n2)
            continue

        # Check distance to all selected vectors
        sel_arr = np.array(selected_vecs, dtype=np.int64)
        diffs = sel_arr - v  # shape (m, D)
        dists2 = np.sum(diffs * diffs, axis=1)
        min_dist2 = int(np.min(dists2))

        if min_dist2 >= new_max_norm2:
            selected_idx.append(idx)
            selected_vecs.append(v)
            current_max_norm2 = new_max_norm2

    # If we ran out of time, return what we have
    if time.time() - t0 > DEADLINE:
        if selected_vecs:
            return np.array(selected_vecs, dtype=np.int64)
        else:
            # Fallback: return the original simple construction
            vectors = []
            for i in range(D):
                for j in range(i + 1, D):
                    for si in (1, -1):
                        for sj in (1, -1):
                            v = [0] * D
                            v[i], v[j] = si, sj
                            vectors.append(v)
            return np.array(vectors, dtype=np.int64)

    if not selected_vecs:
        # Fallback
        vectors = []
        for i in range(D):
            for j in range(i + 1, D):
                for si in (1, -1):
                    for sj in (1, -1):
                        v = [0] * D
                        v[i], v[j] = si, sj
                        vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return np.array(selected_vecs, dtype=np.int64)
