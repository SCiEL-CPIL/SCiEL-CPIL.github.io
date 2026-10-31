"""Kissing number in dimension 11 - MIS on conflict graph of norm-2 vectors."""
import numpy as np
import time
import sys
from itertools import combinations, product
from collections import defaultdict

D = 11
MAX_TIME = 110.0  # Safety margin before 150s kill


def construct():
    start_time = time.time()

    # Generate all integer vectors in R^11 with exactly 4 nonzero entries, each ±1.
    # Count: C(11,4) * 2^4 = 330 * 16 = 5280 vectors.
    # Each has norm sqrt(4) = 2.
    #
    # The kissing condition: min pairwise distance >= max norm.
    # If all vectors have norm 2, we need min pairwise distance >= 2.
    # ||u - v|| >= 2 for all pairs.
    # ||u - v||^2 = 4 + 4 - 2*u.v = 8 - 2*u.v >= 4
    # u.v <= 2
    #
    # For two weight-4 vectors, u.v can be:
    # - If they share k coordinates: u.v = sum of products over shared coords
    #   For each shared coordinate, product is ±1
    #   So u.v ranges from -k to k in steps of 2
    # - If they share 0 coordinates: u.v = 0
    # - If they share 1 coordinate: u.v = ±1
    # - If they share 2 coordinates: u.v = -2, 0, or 2
    # - If they share 3 coordinates: u.v = -3, -1, 1, or 3
    # - If they share 4 coordinates: u.v = -4, -2, 0, 2, or 4
    #
    # Conflict: u.v > 2, i.e., u.v >= 3
    # This happens when they share 3 or 4 coordinates with aligned signs.

    # Generate all vectors
    vectors = []
    support_list = []
    for combo in combinations(range(D), 4):
        for signs in product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int8)
            for idx, s in zip(combo, signs):
                v[idx] = s
            vectors.append(v)
            support_list.append(combo)

    n = len(vectors)  # 5280
    vectors = np.array(vectors)  # shape (n, D)

    # Compute dot products
    dots = vectors @ vectors.T  # (n, n) matrix of dot products

    # Conflict graph: edge between i and j if dots[i,j] > 2
    conflict = (dots > 2).astype(bool)
    np.fill_diagonal(conflict, False)

    # Degrees
    degrees = conflict.sum(axis=1)

    # Greedy MIS with vectorized operations
    def greedy_mis(conflict, order):
        n = len(conflict)
        selected = []
        blocked = np.zeros(n, dtype=bool)
        for v in order:
            if not blocked[v]:
                selected.append(v)
                blocked |= conflict[v]
        return selected

    # Try multiple orderings
    rng = np.random.RandomState(42)
    
    best_set = None
    best_size = 0

    # Ordering 1: by degree (ascending - prefer low degree)
    order_by_degree = np.argsort(degrees)
    selected = greedy_mis(conflict, order_by_degree)
    best_size = len(selected)
    best_set = selected

    # Ordering 2: random
    order_random = rng.permutation(n)
    selected = greedy_mis(conflict, order_random)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected

    # Ordering 3: by degree (descending)
    order_by_degree_desc = np.argsort(-degrees)
    selected = greedy_mis(conflict, order_by_degree_desc)
    if len(selected) > best_size:
        best_size = len(selected)
        best_set = selected

    # Try multiple random restarts
    for restart in range(20):
        if time.time() - start_time > MAX_TIME - 30:
            break
        order_r = rng.permutation(n)
        selected_r = greedy_mis(conflict, order_r)
        if len(selected_r) > best_size:
            best_size = len(selected_r)
            best_set = selected_r

    # Local search: try to improve by removing a vertex and adding multiple others
    if best_set is not None:
        selected_arr = np.zeros(n, dtype=bool)
        selected_arr[best_set] = True

        # Try: for each non-selected vertex v, if it conflicts with k selected vertices,
        # we could remove those k and add v. Net change: 1 - k.
        # Only beneficial if k = 0 (add) or we find a better swap.

        # Try to add vertices by removing conflicting ones
        # For each non-selected v, find its conflicts with selected
        non_selected = np.where(~selected_arr)[0]
        
        for v in non_selected[:1000]:  # Limit for time
            if time.time() - start_time > MAX_TIME - 20:
                break
            conflicts_with_selected = np.where(conflict[v] & selected_arr)[0]
            if len(conflicts_with_selected) == 0:
                # Can add v
                selected_arr[v] = True
                best_size += 1
                best_set = np.where(selected_arr)[0]
            elif len(conflicts_with_selected) == 1:
                # Try swap: remove 1, add 1 (net 0) - skip for now
                pass

    # Convert to final answer
    if best_set is not None and len(best_set) > 0:
        result = vectors[best_set].astype(np.int64)
    else:
        # Fallback: just use a single vector
        result = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int64)

    # Verify
    if len(result) > 1:
        norms = np.sqrt(np.sum(result**2, axis=1))
        max_norm = np.max(norms)
        # Check pairwise distances
        valid = True
        # For efficiency, check all pairs if small, sample if large
        if len(result) <= 300:
            for i in range(len(result)):
                for j in range(i+1, len(result)):
                    d = np.sqrt(np.sum((result[i] - result[j])**2))
                    if d < max_norm - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        else:
            # Sample check
            idx = rng.choice(len(result), size=min(300, len(result)), replace=False)
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    d = np.sqrt(np.sum((result[idx[i]] - result[idx[j]])**2))
                    if d < max_norm - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
        
        if not valid:
            # Fall back to a known valid small set
            # Use weight-2 vectors with disjoint supports
            seed_vecs = []
            for i in range(D):
                for j in range(i+1, D):
                    v = np.zeros(D, dtype=np.int64)
                    v[i] = 1
                    v[j] = 1
                    seed_vecs.append(v)
            result = np.array(seed_vecs, dtype=np.int64)

    return result
