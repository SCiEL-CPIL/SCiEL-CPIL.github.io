"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    N = 100
    target = 200

    def compute_reachable_sums(elements):
        """Compute boolean array of reachable subset sums up to target using numpy."""
        reachable = np.zeros(target + 1, dtype=bool)
        reachable[0] = True
        for x in elements:
            if x > target:
                continue
            reachable[x:] |= reachable[:-x]
        return reachable

    def has_subset_sum_200(elements):
        """Return True if any subset of elements sums to exactly target."""
        reachable = np.zeros(target + 1, dtype=bool)
        reachable[0] = True
        for x in elements:
            if x > target:
                continue
            reachable[x:] |= reachable[:-x]
        return bool(reachable[target])

    def find_subset_sum_fast(elements):
        """Find a subset of elements summing to target, return it or None.
        Uses array-based DP for speed."""
        dp = [-1] * (target + 1)
        dp[0] = -1  # base case: sum 0 reachable with no elements
        parent = {}  # (sum, elem_idx) -> prev_sum
        for i, x in enumerate(elements):
            for s in range(target - x, -1, -1):
                if dp[s] != -1 and dp[s + x] == -1:
                    dp[s + x] = i
                    parent[(s + x, i)] = s
            if dp[target] != -1:
                subset = []
                cur = target
                while cur > 0:
                    i = dp[cur]
                    subset.append(elements[i])
                    cur = parent[(cur, i)]
                return subset
        return None

    def greedy_largest_first():
        selected = []
        reachable = {0}
        for x in range(N, 0, -1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable

    def greedy_smallest_first():
        selected = []
        reachable = {0}
        for x in range(1, N + 1):
            new_sums = {s + x for s in reachable if s + x <= target}
            if target not in new_sums:
                selected.append(x)
                reachable |= new_sums
        return selected, reachable

    def count_subset_sums(elements):
        """Count number of subsets of elements summing to exactly target.
        Returns (count, ways array) where ways[s] = # subsets summing to s."""
        ways = np.zeros(target + 1, dtype=np.int64)
        ways[0] = 1
        for x in elements:
            if x > target:
                continue
            ways[x:] += ways[:-x]
        return int(ways[target]), ways

    def degree_guided_remove(remaining, max_iter=1000):
        """Iteratively remove the highest-degree element (most target-summing
        subsets it participates in). Degree of element x = # target-summing subsets
        of remaining containing x = # subsets of (remaining - x) summing to target-x."""
        remaining = list(remaining)
        removed = set()
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            if not has_subset_sum_200(remaining):
                break
            # Compute degree for each element in remaining
            best_elem = None
            best_deg = -1
            # Compute degrees: for each x in remaining, count subsets of (remaining - x) summing to target - x
            # To be efficient, precompute ways for remaining
            # But that's O(n * target) per element which is expensive.
            # Instead, use a smarter approach: compute ways for remaining once,
            # then degree of x = ways_without_x[target - x].
            # But computing ways_without_x for each x is O(n * target) each.
            # Alternative: use the fact that ways[target] = sum over x in remaining of (subsets containing x that sum to target)
            # Actually, a cleaner approach: for each candidate x, count = # subsets of (remaining - x) summing to (target - x).
            # We can compute this by DP on remaining - x, but that's expensive.
            #
            # Simpler efficient heuristic: use the find_subset_sum_fast to get ONE subset,
            # then for each element in that subset, estimate its degree by how "central" it is.
            # But the true degree requires counting.
            #
            # Compromise: compute degrees only for elements in a found subset (small set),
            # using a targeted DP for each.
            subset = find_subset_sum_fast(remaining)
            if subset is None:
                break
            # For each element in the subset, compute its degree
            for x in set(subset):
                # Count subsets of (remaining - x) summing to target - x
                if target - x < 0 or target - x > target:
                    deg = 0
                else:
                    # DP on remaining without x
                    ways = np.zeros(target + 1, dtype=np.int64)
                    ways[0] = 1
                    for y in remaining:
                        if y == x:
                            continue
                        if y > target:
                            continue
                        ways[y:] += ways[:-y]
                    deg = int(ways[target - x])
                if deg > best_deg:
                    best_deg = deg
                    best_elem = x
            if best_elem is None:
                # Fallback: remove max of subset
                best_elem = max(subset)
            removed.add(best_elem)
            remaining = [x for x in remaining if x != best_elem]
        return remaining

    def iterative_remove_optimal(removal_strategy='critical', max_iter=500):
        """Iteratively remove elements to break all subset-sum-200 paths."""
        remaining = list(range(1, N + 1))
        removed = set()
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            subset = find_subset_sum_fast(remaining)
            if subset is None:
                break
            if removal_strategy == 'largest':
                to_remove = max(subset)
            elif removal_strategy == 'smallest':
                to_remove = min(subset)
            elif removal_strategy == 'random':
                to_remove = random.choice(subset)
            elif removal_strategy == 'critical':
                sorted_subset = sorted(subset)
                mid = len(sorted_subset) // 2
                to_remove = sorted_subset[mid]
            else:
                to_remove = max(subset)
            removed.add(to_remove)
            remaining = [x for x in remaining if x != to_remove]
        return remaining

    # Greedy from large to small:
    selected, reachable = greedy_largest_first()
    best_selected = list(selected)
    best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Try greedy from small to large:
    sel2, reach2 = greedy_smallest_first()
    if len(sel2) > best_size:
        best_selected = list(sel2)
        best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Try different removal strategies
    for strategy in ['largest', 'smallest', 'critical']:
        if time.time() > deadline - 3:
            break
        cand = iterative_remove_optimal(strategy, max_iter=300)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Degree-guided removal (the new hypergraph-inspired heuristic)
    if time.time() < deadline - 5:
        cand = degree_guided_remove(list(range(1, N + 1)), max_iter=200)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Random restarts with random removal order
    random.seed(42)
    for trial in range(200):
        if time.time() > deadline - 3:
            break
        cand = iterative_remove_optimal('random', max_iter=200)
        if len(cand) > best_size:
            best_selected = list(cand)
            best_size = len(best_selected)

    if time.time() > deadline:
        return best_selected

    # Local search: try to add back removed elements
    all_elements = set(range(1, N + 1))
    current_set = set(best_selected)

    def can_add_element(x, current_reachable):
        if x > target:
            return False
        if 200 - x <= target and current_reachable[200 - x]:
            return False
        return True

    current_reachable = compute_reachable_sums(list(current_set))

    for _ in range(100):
        if time.time() > deadline - 2:
            break
        improved = False
        candidates = list(all_elements - current_set)
        random.shuffle(candidates)
        for x in candidates:
            if time.time() > deadline - 2:
                break
            if can_add_element(x, current_reachable):
                current_set.add(x)
                current_reachable[x:] |= current_reachable[:-x]
                improved = True
                break
        if not improved:
            break

    if len(current_set) > best_size:
        best_selected = list(current_set)
        best_size = len(best_selected)

    # Second pass local search with different ordering
    current_set = set(best_selected)
    current_reachable = compute_reachable_sums(list(current_set))

    for _ in range(100):
        if time.time() > deadline - 1:
            break
        improved = False
        candidates = sorted(all_elements - current_set, reverse=True)
        for x in candidates:
            if time.time() > deadline - 1:
                break
            if can_add_element(x, current_reachable):
                current_set.add(x)
                current_reachable[x:] |= current_reachable[:-x]
                improved = True
                break
        if not improved:
            break

    if len(current_set) > best_size:
        best_selected = list(current_set)
        best_size = len(best_selected)

    # Final verification: ensure no subset sums to 200
    # If somehow invalid, fall back to a known-safe greedy result
    if has_subset_sum_200(best_selected):
        # Fallback: use greedy_largest_first which is guaranteed valid by construction
        safe_selected, _ = greedy_largest_first()
        if len(safe_selected) > 0:
            best_selected = safe_selected
            best_size = len(best_selected)

    # Ensure nonempty
    if not best_selected:
        best_selected = [1]
        best_size = 1

    return best_selected
# EVOLVE-BLOCK-END
