"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
D = 11

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    deadline = time.time() + 140.0  # 10s safety margin before 150s kill
    t_start = time.time()

    # We seek nonzero integer vectors in R^11 such that:
    # min_{i != j} ||v_i - v_j|| >= max_i ||v_i||
    #
    # Strategy: Greedy set packing over a large candidate pool.
    # Candidate pool: all integer vectors with entries in {-2, -1, 0, 1, 2}
    # and norm^2 in a small set. We prioritize vectors with the same norm^2
    # to make the constraint uniform: if all have norm^2 = R, then we need
    # ||v_i - v_j||^2 >= R for all i != j.
    #
    # For norm^2 = 4 (e.g., two 2's, or one 2 and two 1's, etc.), the constraint
    # is ||v_i - v_j||^2 >= 4.
    #
    # Actually, let's think more carefully. The condition is:
    #   min pairwise dist >= max norm
    # If all vectors have the same norm R (i.e., ||v||^2 = R for all), then we need
    #   ||v_i - v_j||^2 >= R  for all i != j
    #
    # Let's generate candidates with norm^2 = 4:
    #   - Two entries are +-2, rest 0: C(11,2)*4 = 220 vectors
    #   - One entry is +-2, two entries are +-1: C(11,1)*C(10,2)*4*4 = 11*45*16 = 7920
    #   - Four entries are +-1: C(11,4)*16 = 330*16 = 5280
    # Total: 220 + 7920 + 5280 = 13420 candidates with norm^2 = 4
    #
    # We need pairwise squared distance >= 4.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i.v_j = 8 - 2*v_i.v_j
    # So we need 8 - 2*v_i.v_j >= 4  =>  v_i.v_j <= 2
    #
    # This is a maximum independent set problem in a graph where edges connect
    # pairs with dot product > 2. We'll use a greedy approach with randomization.

    # Generate all candidates with norm^2 = 4
    candidates = []
    import itertools

    # Type A: two +-2 entries
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = np.zeros(D, dtype=np.int64)
                v[i] = 2 * si
                v[j] = 2 * sj
                candidates.append(v)

    # Type B: one +-2, two +-1 entries
    for i in range(D):
        for j, k in itertools.combinations([x for x in range(D) if x != i], 2):
            for si in (1, -1):
                for sj in (1, -1):
                    for sk in (1, -1):
                        v = np.zeros(D, dtype=np.int64)
                        v[i] = 2 * si
                        v[j] = 1 * sj
                        v[k] = 1 * sk
                        candidates.append(v)

    # Type C: four +-1 entries
    for idxs in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, s in zip(idxs, signs):
                v[idx] = s
            candidates.append(v)

    candidates = np.array(candidates, dtype=np.int64)
    n_cand = len(candidates)

    # Compute dot product matrix
    # dot[i,j] = v_i . v_j
    # We need dot[i,j] <= 2 for all selected i != j
    # And dot[i,i] = 4 for all

    # Precompute dot products in chunks to save memory
    # Actually n_cand ~ 13420, so dot matrix is 13420^2 ~ 180M entries.
    # That's ~1.4 GB as int64. Too much. Let's use int32 or compute on the fly.

    # Better approach: greedy with on-the-fly dot product checks.
    # Use a random order and greedily add vectors that don't conflict.

    rng = np.random.default_rng(42)
    order = rng.permutation(n_cand)

    selected = []
    selected_set = set()

    for idx in order:
        if time.time() > deadline:
            break

        v = candidates[idx]

        # Check against all currently selected vectors
        conflict = False
        for s_idx in selected:
            s = candidates[s_idx]
            dot = np.dot(v, s)
            if dot > 2:
                conflict = True
                break

        if not conflict:
            selected.append(idx)

    # Convert to array
    if len(selected) > 0:
        result = candidates[selected]
    else:
        # Fallback: return the seed solution
        result = []
        for i, j in itertools.combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i], v[j] = si, sj
                    result.append(v)
        result = np.array(result, dtype=np.int64)

    return result
# EVOLVE-BLOCK-END
