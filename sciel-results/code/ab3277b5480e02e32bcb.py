"""New design: farthest-point sampling over a superset of norm-4 integer vectors."""
D = 11

# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _build_candidates():
    """Build a superset of nonzero integer vectors with norm^2 <= 4."""
    cands = []

    # Type 1: unit vectors (norm^2 = 1)
    for i in range(D):
        v = np.zeros(D, dtype=np.int64)
        v[i] = 1
        cands.append(v)

    # Type 2: weight-2 vectors with entries in {+1,-1} (norm^2 = 2)
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i], v[j] = si, sj
                cands.append(v)

    # Type 3: weight-1 with a 2 entry (norm^2 = 4)
    for i in range(D):
        for s in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = s
            cands.append(v)

    return np.array(cands, dtype=np.int64)


def construct():
    deadline = time.time() + 140.0
    candidates = _build_candidates()
    n = len(candidates)

    # norms^2 of each candidate
    norms2 = np.einsum('ij,ij->i', candidates, candidates)

    # Precompute pairwise squared distances matrix (n x n)
    # dist^2 = |a|^2 + |b|^2 - 2 a.b
    dots = candidates @ candidates.T
    dist2 = norms2[:, None] + norms2[None, :] - 2 * dots
    np.maximum(dist2, 0, out=dist2)
    np.fill_diagonal(dist2, 0)

    # Greedy farthest-point sampling:
    # maintain min distance from each candidate to the chosen set.
    chosen = []
    min_dist2 = dist2.copy()  # min distance to chosen set; initially to nothing -> inf
    # We need to pick the first point: pick one with largest norm to set the bar.
    # The constraint: min pairwise dist >= max norm in the chosen set.
    # We'll track the current max norm among chosen.

    # Initialize min_dist2 to infinity (no chosen yet). We'll handle first pick specially.
    min_dist2 = np.full(n, np.inf, dtype=np.int64)
    chosen_set = []
    max_norm2 = 0

    while time.time() < deadline:
        # Pick the candidate with the largest min_dist2 (farthest from chosen set).
        # But we must ensure the constraint holds: the new point's distance to all
        # chosen must be >= max_norm (the max norm of chosen). Also the new point's
        # own norm must be <= the min pairwise distance among chosen (to keep
        # min pairwise dist >= max norm after adding).
        #
        # The constraint after adding point p:
        #   min pairwise dist among (chosen + p) >= max norm among (chosen + p)
        # The min pairwise dist among chosen is currently >= max_norm (invariant).
        # After adding p:
        #   new min pairwise dist = min(old_min_pairwise, min_dist_to_chosen(p))
        #   new max norm = max(max_norm, norm(p))
        # We need new_min_pairwise >= new_max_norm.
        #
        # Since we don't track old_min_pairwise exactly, we enforce a safe condition:
        #   min_dist_to_chosen(p) >= max(max_norm, norm(p))
        # This guarantees new_min_pairwise >= max_norm and >= norm(p), so
        # new_min_pairwise >= new_max_norm.

        # Find candidates satisfying the constraint and pick the farthest.
        # norm(p)^2 = norms2[p]; we need min_dist2[p] >= max(max_norm2, norms2[p]).
        # min_dist2[p] is already the min squared distance to chosen set.
        # Condition: min_dist2[p] >= max(max_norm2, norms2[p])  (squared)

        # Build a score: among feasible candidates, pick max min_dist2.
        # Feasible: min_dist2[p] >= max(max_norm2, norms2[p])
        # For the very first point, min_dist2 is inf, so any is feasible;
        # pick the one with largest norm to maximize the bar.
        if len(chosen_set) == 0:
            idx = int(np.argmax(norms2))
        else:
            feasible = min_dist2 >= np.maximum(max_norm2, norms2)
            if not np.any(feasible):
                break
            # Among feasible, pick max min_dist2
            scores = np.where(feasible, min_dist2, -1)
            idx = int(np.argmax(scores))
            if scores[idx] < 0:
                break

        chosen_set.append(idx)
        # Update max_norm2
        if norms2[idx] > max_norm2:
            max_norm2 = norms2[idx]
        # Update min_dist2 for all candidates
        min_dist2 = np.minimum(min_dist2, dist2[idx])

    if not chosen_set:
        # Fallback: return the seed-like construction
        vectors = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    vectors.append(v)
        return np.array(vectors, dtype=np.int64)

    return candidates[chosen_set]
# EVOLVE-BLOCK-END
