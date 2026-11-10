# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import time


def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional points with integral coordinates
    satisfying: max squared norm < min squared pairwise distance.
    Uses a greedy maximum-clique approach on the D11 minimal vector graph.
    """
    deadline = time.time() + 580  # 580s out of 600s budget

    d = 11
    # Generate D11 minimal vectors: weight-2 vectors
    # Type A: e_i + e_j (positive-positive)
    # Type B: e_i - e_j (positive-negative)
    # Type C: e_i - e_j where i,j are specific signs
    # Actually, D_n minimal vectors have norm 2 and are:
    # For n odd: weight 2 vectors with one +1 and one -1, or two +1s or two -1s
    # D_n = {x in Z^n : sum x_i is even}
    # Minimal vectors of D_n: all weight-2 vectors in Z^n with even sum
    # Weight-2 vectors: exactly two non-zero entries
    # For even sum: either both entries are +1, both -1, or one +1 and one -1
    # But +1 and -1 sums to 0 (even), +1 and +1 sums to 2 (even), -1 and -1 sums to -2 (even)
    # So all weight-2 vectors in Z^n with entries in {-1, 0, 1} are in D_n
    # Count: C(n,2) * 4 = 55 * 4 = 220 for n=11

    points = []
    for i in range(d):
        for j in range(i + 1, d):
            # Four sign combinations for (i, j)
            for si in [1, -1]:
                for sj in [1, -1]:
                    v = np.zeros(d, dtype=np.int64)
                    v[i] = si
                    v[j] = sj
                    points.append(v)

    points = np.array(points, dtype=np.int64)
    n = len(points)

    # Compute dot product matrix
    # D_ij = points[i] . points[j]
    # We need: for all pairs in our set, |D_ij| >= 1? No...
    # Condition: ||x - y||^2 >= max(||x||^2, ||y||^2)
    # Since all points have norm 2: ||x-y||^2 = 2 + 2 - 2*x.y = 4 - 2*x.y >= 2
    # So x.y <= 1
    # And we need points distinct and non-zero (satisfied)
    # Also need: min pairwise squared dist >= max squared norm
    # min ||x-y||^2 >= 2 (max norm squared)
    # ||x-y||^2 = 4 - 2*x.y >= 2 => x.y <= 1
    # So the condition is: for all x != y in S, x . y <= 1

    # Precompute compatibility: edge between i,j if points[i].points[j] <= 1
    D = points @ points.T  # (n, n) dot product matrix

    # Build adjacency: adj[i] = set of j where D[i,j] <= 1 and j != i
    # Actually for greedy clique, we just need to check dot products

    # Greedy max clique heuristic with multiple random orderings
    best_set = []
    best_size = 0

    # Initial greedy: pick vertices in order of lowest degree (most restrictive first? or least?)
    # Actually for max clique, we want to add vertices that are compatible with many others
    # Degree in the compatibility graph = number of j with D[i,j] <= 1

    deg = np.sum(D <= 1, axis=1) - 1  # subtract self (D[i,i]=2 > 1, so self is not in the set anyway)
    # Wait, D[i,i] = 2, so D[i,i] <= 1 is False. So deg[i] = sum(D[i,:] <= 1) already excludes self.
    deg = np.sum(D <= 1, axis=1)

    # Try multiple strategies
    for attempt in range(200):
        if time.time() > deadline:
            break

        if attempt == 0:
            # Greedy by degree (highest degree first - most compatible)
            order = np.argsort(-deg)
        elif attempt == 1:
            # Greedy by degree (lowest degree first)
            order = np.argsort(deg)
        else:
            # Random ordering
            order = np.random.permutation(n)

        current_set = []
        current_indices = set()

        for idx in order:
            if len(current_set) <= best_size:
                # Check if idx is compatible with all in current_set
                # We need D[idx, j] <= 1 for all j in current_set
                if current_set:
                    # Check dot products
                    if not np.all(D[idx, current_set] <= 1):
                        continue
            current_set.append(idx)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    # Try to improve with local search / re-optimization
    # For the best set found, try swapping or adding
    if best_set:
        current_set = best_set[:]
        current_indices = set(current_set)
        # Try to add more points
        for idx in range(n):
            if idx in current_indices:
                continue
            if np.all(D[idx, current_set] <= 1):
                current_set.append(idx)
                current_indices.add(idx)

        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]

    # Final validation
    if not best_set or len(best_set) < 2:
        # Fallback to the seed answer
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

    result = points[best_set]

    # Validate
    k = len(result)
    if k < 2:
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

    # Check no zero vector
    if np.any(np.all(result == 0, axis=1)):
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

    # Check min squared distance >= max squared norm
    max_norm_sq = np.max(np.sum(result ** 2, axis=1))
    min_dist_sq = float('inf')
    for i in range(k):
        for j in range(i + 1, k):
            diff = result[i] - result[j]
            dist_sq = np.sum(diff ** 2)
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq

    if min_dist_sq < max_norm_sq:
        # Invalid, fall back
        return np.array([np.ones(d), -1 * np.ones(d)]).astype(np.int64)

    return np.around(result).astype(np.int64)


# EVOLVE-BLOCK-END
