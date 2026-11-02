"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # We want nonzero integer vectors in R^11 such that
    # min pairwise distance >= max vector norm.
    # Normalizing to norm 2 gives kissing configuration.
    #
    # Strategy: Use weight-4 vectors (exactly 4 nonzero entries, each +/-1).
    # For two weight-4 vectors u, v with norms 2:
    #   ||u - v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 8 - 2<u,v>
    # We need ||u-v|| >= 2 (max norm), so 8 - 2<u,v> >= 4, i.e., <u,v> <= 2.
    #
    # Symmetry breaking: restrict to vectors where the first nonzero coordinate is +1.
    # This reduces the candidate pool by factor of 2 (sign symmetry) and we can
    # further reduce by considering only "canonical" representatives under coordinate
    # permutations, but that's complex. Instead, we'll use a greedy approach on a
    # well-ordered candidate set.
    #
    # Actually, let's think more carefully. The kissing number in dimension 11 is known
    # to be at least 460 (from constructions) and at most 460? Let me recall:
    # Known kissing numbers: d=1:2, d=2:6, d=3:12, d=4:24, d=5:40, d=6:56, d=7:72,
    # d=8:240, d=9:272, d=10:346, d=11:460? Actually for d=11, the known lower bound
    # is around 460 and upper bound is 460? Let me just focus on constructing a good set.
    #
    # For weight-4 vectors with entries in {+1, -1}, the inner product between two such
    # vectors depends on how many coordinates they share with the same sign.
    # If they share k coordinates with same sign and (4-k) with opposite sign in the
    # overlap, and have disjoint support for the rest... actually it's easier to compute directly.
    #
    # Let's generate all weight-4 vectors with first nonzero entry positive, then do greedy selection.

    # Generate candidate vectors: all subsets of 4 coordinates from 11, with signs
    # where the first nonzero (smallest index) is +1.
    candidates = []
    for coords in combinations(range(D), 4):
        # For each choice of signs, ensure first coordinate (smallest index) is +1
        for signs in range(8):  # 2^3 = 8 sign choices for positions 1,2,3 (position 0 fixed to +1)
            v = np.zeros(D, dtype=np.int64)
            v[coords[0]] = 1
            for i in range(1, 4):
                if (signs >> (i-1)) & 1:
                    v[coords[i]] = -1
                else:
                    v[coords[i]] = 1
            candidates.append(v)

    # Sort candidates for deterministic ordering
    candidates = np.array(candidates, dtype=np.int64)

    # Greedy min-degree selection: repeatedly pick the vector that has the fewest
    # conflicts with already-selected vectors, or start with a vector and add compatible ones.
    # Actually, let's do a greedy independent set: iterate through candidates in order,
    # add a vector if it's compatible with all already-selected vectors.
    #
    # But a better approach: sort by some heuristic and greedily add.
    # Let's try: for each candidate in order, check if it's compatible with all selected.
    # Two vectors are compatible if ||u - v||^2 >= max(||u||^2, ||v||^2) = 8.
    # Since all have norm^2 = 8, we need ||u-v||^2 >= 8, i.e., 8 - 2<u,v> >= 8, i.e., <u,v> <= 0.
    # Wait, that's stricter than I thought. Let me re-derive.
    #
    # The condition is: min pairwise distance >= max vector norm.
    # All vectors have norm 2 (norm^2 = 4? No wait, weight-4 with entries +/-1: norm^2 = 4, norm = 2).
    # So max norm = 2. We need ||u - v|| >= 2 for all pairs.
    # ||u-v||^2 = ||u||^2 + ||v||^2 - 2<u,v> = 4 + 4 - 2<u,v> = 8 - 2<u,v>
    # Need 8 - 2<u,v> >= 4, so <u,v> <= 2.
    #
    # So the condition is <u,v> <= 2 for all distinct pairs.

    selected = []
    selected_sq = np.array([])  # stack of selected vectors for batch inner product computation

    for i in range(len(candidates)):
        if time.time() > deadline:
            break
        v = candidates[i]
        if len(selected) == 0:
            selected.append(v)
            selected_sq = v.reshape(1, -1).astype(np.float64)
        else:
            # Check compatibility with all selected
            inner_products = selected_sq @ v
            if np.all(inner_products <= 2):
                selected.append(v)
                selected_sq = np.vstack([selected_sq, v.reshape(1, -1).astype(np.float64)])

    result = np.array(selected, dtype=np.int64)
    if len(result) == 0:
        # Fallback: return the seed solution
        vectors = []
        for i, j in combinations(range(D), 2):
            for si in (1, -1):
                for sj in (1, -1):
                    vec = [0] * D
                    vec[i], vec[j] = si, sj
                    vectors.append(vec)
        result = np.array(vectors, dtype=np.int64)

    return result

# EVOLVE-BLOCK-END
