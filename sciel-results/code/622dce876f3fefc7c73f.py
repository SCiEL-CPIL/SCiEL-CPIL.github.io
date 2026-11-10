# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a maximum-size subset of Z/205Z with no two elements differing by a
    nonzero square modulo 205.

    Strategy:
      1. Build the forbidden-difference set F (nonzero quadratic residues mod 205).
      2. Use a randomized greedy construction with multiple random orderings to
         obtain diverse feasible subsets.
      3. Apply local search (2-opt style swaps and insertions) to improve size.
      4. Track the best feasible subset found and return it.

    Returns:
        np.ndarray of distinct residues in 0..204, no pair has a forbidden difference.
    """
    m = 205
    deadline = time.time() + 580  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)

    # Precompute for each pair (a,b) whether they are compatible
    # compat[a][b] = True if (a-b) mod m not in F and (b-a) mod m not in F
    # Actually since F is symmetric (if d in F then (-d) mod m may or not be in F)
    # Let's just check both directions
    def is_compatible(a, b):
        d1 = (a - b) % m
        d2 = (b - a) % m
        return d1 not in forbidden and d2 not in forbidden

    # Precompute compatibility matrix for speed
    # compat[i][j] = 1 if i and j can coexist, 0 otherwise
    compat = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        for j in range(i + 1, m):
            if is_compatible(i, j):
                compat[i][j] = 1
                compat[j][i] = 1

    def is_valid_subset(subset):
        """Check if subset is a valid square-difference-free set."""
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset[i], subset[j]
                if not compat[a, b]:
                    return False
        return True

    def greedy_random_order(order):
        """Greedy construction following a given order."""
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                if not compat[candidate, s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    def local_search_improve(subset, max_iter=5000):
        """Try to improve subset by adding elements or swapping."""
        subset = list(subset)
        selected_set = set(subset)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            # Try adding any unselected element
            for cand in range(m):
                if cand in selected_set:
                    continue
                ok = True
                for s in subset:
                    if not compat[cand, s]:
                        ok = False
                        break
                if ok:
                    subset.append(cand)
                    selected_set.add(cand)
                    improved = True
                    break
            if improved:
                continue
            # Try swapping: remove one element, add another
            for idx_remove in range(len(subset)):
                if time.time() >= deadline:
                    break
                removed = subset[idx_remove]
                remaining = [s for i, s in enumerate(subset) if i != idx_remove]
                remaining_set = set(remaining)
                best_add = None
                for cand in range(m):
                    if cand in remaining_set or cand == removed:
                        continue
                    ok = True
                    for s in remaining:
                        if not compat[cand, s]:
                            ok = False
                            break
                    if ok:
                        best_add = cand
                        break
                if best_add is not None:
                    subset = remaining + [best_add]
                    selected_set = set(subset)
                    improved = True
                    break
        return subset

    # Multiple random restarts with greedy + local search
    best_subset = []
    best_size = 0

    # First, try a deterministic greedy in natural order
    init_subset = greedy_random_order(list(range(m)))
    init_subset = local_search_improve(init_subset)
    if len(init_subset) > best_size:
        best_size = len(init_subset)
        best_subset = list(init_subset)

    # Randomized greedy with local search
    num_restarts = 200
    for r in range(num_restarts):
        if time.time() >= deadline:
            break
        random.seed(r * 12345 + 67890)
        order = list(range(m))
        random.shuffle(order)
        subset = greedy_random_order(order)
        subset = local_search_improve(subset, max_iter=2000)
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = list(subset)

    # Additional local search passes on the best subset found
    if time.time() < deadline - 30:
        improved = True
        while improved and time.time() < deadline - 30:
            improved = False
            best_subset = local_search_improve(best_subset, max_iter=10000)
            if len(best_subset) > best_size:
                best_size = len(best_subset)
                improved = True

    # Final validation
    if not is_valid_subset(best_subset):
        # Fallback to a known valid small set
        best_subset = [0]
        best_size = 1

    return np.array(best_subset, dtype=np.int64)


# EVOLVE-BLOCK-END
