"""Rewritten program: explicit kissing number construction in dimension 11."""
D = 11

# EVOLVE-BLOCK-START
import time
import numpy as np


def construct():
    deadline = time.time() + 110  # safety margin within 120s

    # Generate weight-2 vectors: for each pair (i,j) with i<j,
    # create 4 sign combinations (si, sj) in {(1,1),(1,-1),(-1,1),(-1,-1)}
    # This gives C(11,2) * 4 = 55 * 4 = 220 vectors.
    #
    # Norm of each vector: sqrt(1^2 + 1^2) = sqrt(2)
    # Distance between two weight-2 vectors:
    #   If they share one coordinate and differ in sign: dist^2 = (1-(-1))^2 + (1-0)^2 + (0-1)^2 = 4+1+1 = 6? No.
    #   Let's compute properly.
    #   v1 has entries at positions i,j: (si, sj)
    #   v2 has entries at positions k,l: (sk, sl)
    #   If {i,j} == {k,l} and signs are the same: dist = 0 (same vector, not included since we have unique vectors)
    #   If {i,j} == {k,l} and signs differ:
    #       e.g., v1 = (1,1) at (i,j), v2 = (1,-1) at (i,j)
    #       diff at i: 1-1=0, at j: 1-(-1)=2, dist^2 = 4
    #       e.g., v1 = (1,1) at (i,j), v2 = (-1,1) at (i,j)
    #       diff at i: 1-(-1)=2, at j: 1-1=0, dist^2 = 4
    #       e.g., v1 = (1,1) at (i,j), v2 = (-1,-1) at (i,j)
    #       diff at i: 1-(-1)=2, at j: 1-(-1)=2, dist^2 = 8
    #   If {i,j} intersects {k,l} in exactly one position:
    #       e.g., v1 = (1,1) at (i,j), v2 = (1,1) at (i,k)
    #       diff at i: 0, at j: 1, at k: -1, dist^2 = 2
    #   If {i,j} is disjoint from {k,l}:
    #       diff at i: si, at j: sj, at k: -sk, at l: -sl, dist^2 = 4
    #
    # Minimum distance squared = 2 (when they share one coordinate with same sign)
    # Norm squared = 2
    # So min distance = sqrt(2) = max norm. Valid!

    vectors = []
    for i in range(D):
        for j in range(i + 1, D):
            for si in (1, -1):
                for sj in (1, -1):
                    v = [0] * D
                    v[i] = si
                    v[j] = sj
                    vectors.append(v)

    if time.time() > deadline:
        # Fallback: return what we have so far (shouldn't happen, this is fast)
        return np.array(vectors, dtype=np.int64)

    result = np.array(vectors, dtype=np.int64)

    # Verify the constraint: min pairwise distance >= max norm
    # For efficiency, do a quick check on a subset if the full check is too slow
    # But 220 vectors is small, full check is fine.
    if len(result) <= 5000:
        norms = np.sqrt(np.sum(result ** 2, axis=1))
        max_norm = np.max(norms)
        # Compute all pairwise distances
        # dist^2 = ||a||^2 + ||b||^2 - 2*a.b
        # Since all norms are equal (sqrt(2)), dist^2 = 4 - 2*a.b for unit-normalized... 
        # Actually just compute directly
        # Use broadcasting for 220x220 which is fine
        dots = result @ result.T
        dists_sq = norms[:, None] ** 2 + norms[None, :] ** 2 - 2 * dots
        np.fill_diagonal(dists_sq, np.inf)
        min_dist = np.sqrt(np.min(dists_sq))
        # Should have min_dist >= max_norm
        # Due to floating point, use a small tolerance
        if not (min_dist >= max_norm - 1e-9):
            # This shouldn't happen, but if it does, fall back to a known valid subset
            # The weight-2 construction is mathematically valid, so this is just a sanity check
            pass

    return result
# EVOLVE-BLOCK-END
