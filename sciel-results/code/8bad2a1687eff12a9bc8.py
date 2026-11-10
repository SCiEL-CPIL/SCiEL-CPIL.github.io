# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2

    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)

    # Precompute a boolean array for fast lookup
    forbidden_arr = np.zeros(m, dtype=bool)
    for f in forbidden:
        forbidden_arr[f] = True

    # Conflict matrix: conflict[i][j] is True if (i-j) mod m is in forbidden
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d != 0 and forbidden_arr[d]:
                conflict[i][j] = True

    def is_valid(subset):
        """Check if subset is a valid independent set."""
        s = len(subset)
        if s == 0:
            return True
        for i in range(s):
            for j in range(i + 1, s):
                d1 = (subset[i] - subset[j]) % m
                d2 = (subset[j] - subset[i]) % m
                if forbidden_arr[d1] or forbidden_arr[d2]:
                    return False
        return True

    def greedy_random(order, deadline):
        """Greedy construction given a specific order of candidates."""
        selected = []
        selected_set = set()
        for candidate in order:
            if time.time() > deadline:
                break
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if forbidden_arr[d1] or forbidden_arr[d2]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected

    def local_search(subset, deadline, max_iter=5000):
        """
        Local search: try to add elements, or swap elements to free up space.
        Returns the best subset found.
        """
        subset = list(subset)
        best = list(subset)
        best_size = len(subset)

        for _ in range(max_iter):
            if time.time() > deadline:
                break

            made_change = False

            # Try adding elements
            for candidate in range(m):
                if candidate in subset:
                    continue
                ok = True
                for s in subset:
                    d1 = (candidate - s) % m
                    d2 = (s - candidate) % m
                    if forbidden_arr[d1] or forbidden_arr[d2]:
                        ok = False
                        break
                if ok:
                    subset.append(candidate)
                    if len(subset) > best_size:
                        best = list(subset)
                        best_size = len(subset)
                    made_change = True
                    break

            if made_change:
                continue

            # Try swaps: remove one element, try to add two
            for remove_idx in range(len(subset)):
                removed = subset[remove_idx]
                temp_subset = subset[:remove_idx] + subset[remove_idx + 1:]

                # Try to add elements to temp_subset
                added_count = 0
                additions = []
                for candidate in range(m):
                    if candidate in temp_subset or candidate == removed:
                        continue
                    ok = True
                    for s in temp_subset:
                        d1 = (candidate - s) % m
                        d2 = (s - candidate) % m
                        if forbidden_arr[d1] or forbidden_arr[d2]:
                            ok = False
                            break
                    if ok:
                        temp_subset.append(candidate)
                        additions.append(candidate)
                        added_count += 1
                        if added_count >= 2:
                            break

                if added_count >= 2:
                    # This swap improves size
                    if len(temp_subset) > best_size:
                        best = list(temp_subset)
                        best_size = len(temp_subset)
                    subset = temp_subset
                    made_change = True
                    break

                # Try adding just one element (keep size same, maybe enables future)
                if added_count == 1:
                    # Only accept if it helps in some heuristic way; for now skip to avoid cycling
                    pass

            if not made_change:
                # No improvement possible with current moves; break
                break

        return best

    # Main search loop
    deadline = time.time() + 580  # 580 seconds to be safe
    best_subset = []
    best_size = 0

    # First, do a few deterministic greedy passes with different orderings
    # Order 1: natural
    candidates = list(range(m))
    s = greedy_random(candidates, deadline)
    if len(s) > best_size:
        best_subset = s
        best_size = len(s)

    # Order 2: reversed
    candidates = list(range(m - 1, -1, -1))
    s = greedy_random(candidates, deadline)
    if len(s) > best_size:
        best_subset = s
        best_size = len(s)

    # Multiple randomized greedy + local search
    random.seed(42)  # Reproducible
    num_trials = 200
    for trial in range(num_trials):
        if time.time() > deadline:
            break

        # Random order
        candidates = list(range(m))
        random.shuffle(candidates)

        s = greedy_random(candidates, deadline)
        if len(s) > best_size:
            best_subset = s
            best_size = len(s)

        # Local search from this starting point
        s2 = local_search(s, deadline, max_iter=2000)
        if len(s2) > best_size:
            best_subset = s2
            best_size = len(s2)

        # Also try local search from the current best
        if trial % 10 == 0:
            s3 = local_search(best_subset, deadline, max_iter=1000)
            if len(s3) > best_size:
                best_subset = s3
                best_size = len(s3)

    # Final verification
    if not is_valid(best_subset):
        # Fallback to a simple valid set
        best_subset = [0]

    return np.array(best_subset, dtype=np.int64)


# EVOLVE-BLOCK-END
