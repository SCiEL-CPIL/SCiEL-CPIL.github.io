"""Construct a set of distinct integers in 1..256 where subset sums
with different cardinalities are always distinct.
Uses direct construction of high-value arithmetic progressions plus
targeted local improvement."""
import time

def construct():
    deadline = time.time() + 55  # safety margin before 60s limit

    def is_valid(sel):
        """Check if a sorted list of distinct ints satisfies the constraint:
        for all k from 1 to n-1, max sum of (k-1) elements < min sum of k elements."""
        n = len(sel)
        if n == 0:
            return True
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + sel[i]
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + sel[i]

        for k in range(1, n):
            # max sum of (k-1) elements: sum of largest (k-1) elements
            if k - 1 == 0:
                max_sum_km1 = 0
            else:
                max_sum_km1 = suffix[n - (k - 1)]
            # min sum of k elements: sum of smallest k elements
            min_sum_k = prefix[k]
            if max_sum_km1 >= min_sum_k:
                return False
        return True

    def try_construct(n, base, step):
        """Try to construct a set of size n as an arithmetic progression
        starting at base with common difference step, all values in 1..256.
        Returns the list if valid, else None."""
        if n <= 0:
            return []
        # The set is: base, base+step, base+2*step, ..., base+(n-1)*step
        # All must be in 1..256
        if base < 1 or base < 1:
            return None
        last = base + (n - 1) * step
        if last > 256:
            return None
        sel = [base + i * step for i in range(n)]
        if is_valid(sel):
            return sel
        return None

    def local_search(initial, candidates, max_iterations=50000):
        """Try to improve the set by swapping elements or adding new ones."""
        current = list(initial)
        current_set = set(current)
        best = list(current)
        best_len = len(current)
        start_time = time.time()

        for iteration in range(max_iterations):
            if time.time() > deadline:
                break

            improved = False

            # Try adding a candidate
            for cand in candidates:
                if cand in current_set:
                    continue
                new_sel = sorted(current + [cand])
                if is_valid(new_sel):
                    current = new_sel
                    current_set.add(cand)
                    best = list(current)
                    best_len = len(current)
                    improved = True
                    break

            if improved:
                continue

            # Try swapping: remove one element and add another
            if len(current) > 0:
                for i in range(len(current)):
                    for cand in candidates:
                        if cand in current_set:
                            continue
                        if cand == current[i]:
                            continue
                        new_sel = current[:i] + current[i+1:] + [cand]
                        new_sel = sorted(new_sel)
                        if is_valid(new_sel):
                            # Only accept if it helps (same size but maybe better position for further additions)
                            # We'll accept swaps that maintain size; the goal is to enable further additions
                            current = new_sel
                            current_set = set(current)
                            if len(current) > best_len:
                                best = list(current)
                                best_len = len(current)
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                # No improvement possible with current candidates
                break

        return best

    # Strategy: try to construct sets of increasing size using high-value APs
    # For n elements, the tightest constraint is usually around k = n/2
    # A high base value makes the gaps work in our favor

    best_set = []
    best_size = 0

    # Try sizes from 35 down to 20, and for each size try various base/step combos
    for n in range(35, 15, -1):
        if time.time() > deadline - 5:
            break

        # For a given n, we want base to be as large as possible
        # and step to be as small as possible (but >= 1)
        # last = base + (n-1)*step <= 256
        # So base <= 256 - (n-1)*step
        # We want base >= 1

        # Try step = 1 first (consecutive integers), then step = 2, etc.
        for step in range(1, 5):
            max_base = 256 - (n - 1) * step
            if max_base < 1:
                continue

            # Try the largest possible base first
            for base in range(max_base, max(1, max_base - 20), -1):
                if time.time() > deadline - 2:
                    break
                sel = try_construct(n, base, step)
                if sel is not None:
                    if len(sel) > best_size:
                        best_set = sel
                        best_size = len(sel)
                    # Try local search to see if we can add more
                    if time.time() < deadline - 3:
                        candidates = [c for c in range(1, 257) if c not in set(sel)]
                        improved = local_search(sel, candidates)
                        if len(improved) > best_size:
                            best_set = improved
                            best_size = len(improved)
                    break  # Found valid for this n,step; move to next n
            if best_size >= n:
                break
        if best_size >= n:
            break

    # If we haven't found a large set, use greedy backtracking as fallback
    if best_size < 25:
        def greedy_construct():
            sel = []
            for cand in range(256, 0, -1):
                new_sel = sorted(sel + [cand])
                if is_valid(new_sel):
                    sel = new_sel
            return sel

        greedy = greedy_construct()
        if len(greedy) > best_size:
            best_set = greedy
            best_size = len(best_set)

        # Try local search on greedy result
        if time.time() < deadline - 2 and best_size < 30:
            candidates = [c for c in range(1, 257) if c not in set(best_set)]
            improved = local_search(best_set, candidates)
            if len(improved) > best_size:
                best_set = improved

    # Final validation
    best_set = sorted(set(best_set))
    # Ensure all are ints in 1..256
    best_set = [x for x in best_set if isinstance(x, int) and 1 <= x <= 256]

    return best_set
