# EVOLVE-BLOCK-START
import numpy as np
import time


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses a combination of greedy construction and local search (2-opt, 3-opt) to maximize size.
    """
    m = 205
    deadline = time.time() + 570  # Safety margin before 600s limit

    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = (x * x) % m
        if sq != 0:
            forbidden.add(sq)

    # Precompute adjacency: for each pair (i, j), check if difference is forbidden
    # adj[i][j] = True if (i-j) mod m is in forbidden
    adj = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden:
                adj[i, j] = True

    def is_valid_subset(subset):
        """Check if subset has no forbidden pairs."""
        s = set(subset)
        for a in s:
            for b in s:
                if a != b and adj[a, b]:
                    return False
        return True

    def count_conflicts(subset):
        """Count number of forbidden pairs in subset."""
        s = set(subset)
        count = 0
        for a in s:
            for b in s:
                if a < b and adj[a, b]:
                    count += 1
        return count

    def greedy_construct():
        """Greedy construction: add elements in order 0..204 if no conflict."""
        selected = []
        for candidate in range(m):
            ok = True
            for s in selected:
                if adj[candidate, s] or adj[s, candidate]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    def local_search_2opt(subset):
        """Try to improve subset by removing one and adding another, or swapping."""
        subset = list(subset)
        improved = True
        best_size = len(subset)

        while improved:
            improved = False
            # Try removing one element and adding another
            for i in range(len(subset)):
                for j in range(m):
                    if j in subset:
                        continue
                    # Try removing subset[i] and adding j
                    new_subset = subset[:i] + subset[i+1:] + [j]
                    new_subset = sorted(new_subset)
                    if count_conflicts(new_subset) == 0:
                        if len(new_subset) > len(subset):
                            subset = new_subset
                            improved = True
                            best_size = len(subset)
                            break
                if improved:
                    break

            # Try swapping: remove one, add another (same size but maybe enables further growth)
            if not improved:
                for i in range(len(subset)):
                    for j in range(m):
                        if j in subset:
                            continue
                        new_subset = subset[:i] + subset[i+1:] + [j]
                        new_subset = sorted(new_subset)
                        if count_conflicts(new_subset) == 0:
                            # Same size, but different composition - accept if it allows future growth
                            # For now, just track it
                            pass

        return subset

    def local_search_3opt(subset):
        """Try to improve by removing up to 2 and adding up to 2."""
        subset = list(subset)
        improved = True

        while improved:
            improved = False
            # Try removing 1 and adding 2
            for i in range(len(subset)):
                for j1 in range(m):
                    if j1 in subset:
                        continue
                    for j2 in range(m):
                        if j2 in subset or j2 == j1:
                            continue
                        new_subset = subset[:i] + subset[i+1:] + [j1, j2]
                        new_subset = sorted(new_subset)
                        if count_conflicts(new_subset) == 0:
                            if len(new_subset) > len(subset):
                                subset = new_subset
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break

            # Try removing 2 and adding 3
            if not improved:
                for i in range(len(subset)):
                    for k in range(i+1, len(subset)):
                        for j1 in range(m):
                            if j1 in subset:
                                continue
                            for j2 in range(m):
                                if j2 in subset or j2 == j1:
                                    continue
                                for j3 in range(m):
                                    if j3 in subset or j3 in (j1, j2):
                                        continue
                                    new_subset = subset[:i] + subset[i+1:k] + subset[k+1:] + [j1, j2, j3]
                                    new_subset = sorted(new_subset)
                                    if count_conflicts(new_subset) == 0:
                                        if len(new_subset) > len(subset):
                                            subset = new_subset
                                            improved = True
                                            break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break

        return subset

    def random_restart_greedy():
        """Greedy with random ordering."""
        import random
        candidates = list(range(m))
        random.shuffle(candidates)
        selected = []
        for candidate in candidates:
            ok = True
            for s in selected:
                if adj[candidate, s] or adj[s, candidate]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected

    # Start with greedy
    best_subset = greedy_construct()
    best_size = len(best_subset)

    # Improve with 2-opt
    improved = local_search_2opt(best_subset)
    if len(improved) > best_size:
        best_subset = improved
        best_size = len(best_subset)

    # Try random restarts
    import random
    random.seed(42)
    start_time = time.time()

    for restart in range(1000):
        if time.time() > deadline:
            break

        # Random greedy
        subset = random_restart_greedy()
        if len(subset) < best_size:
            continue

        # 2-opt
        improved = local_search_2opt(subset)
        if len(improved) > best_size:
            best_subset = improved
            best_size = len(best_subset)

        # 3-opt if we have time
        if time.time() < deadline - 60:
            improved3 = local_search_3opt(best_subset)
            if len(improved3) > best_size:
                best_subset = improved3
                best_size = len(best_subset)

    # Final 2-opt and 3-opt on best
    if time.time() < deadline - 30:
        improved = local_search_2opt(best_subset)
        if len(improved) > best_size:
            best_subset = improved
            best_size = len(improved)

    if time.time() < deadline - 10:
        improved3 = local_search_3opt(best_subset)
        if len(improved3) > best_size:
            best_subset = improved3
            best_size = len(improved3)

    # Verify final result
    if not is_valid_subset(best_subset):
        # Fallback to greedy if something went wrong
        best_subset = greedy_construct()

    return np.array(sorted(best_subset), dtype=np.int64)


# EVOLVE-BLOCK-END
