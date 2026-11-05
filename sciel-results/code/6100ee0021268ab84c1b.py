import time

def construct():
    start = time.time()
    deadline = start + 115.0
    full_mask = (1 << 30) - 1
    max_size = 32

    # Bitmask of all differences up to 60 (differences within 0..120 are <= 120,
    # but we only care about conflicts for diffs <= 60 since we search D <= 60)
    # Actually, for D up to 60, max diff is 60. Let's use 64 bits to be safe.
    # We need to track all differences that have been used to avoid Sidon violations.
    # Max difference in a set with max element D is D. So we need bits 0..D.
    # Let's use a Python int as a bitmask for used differences (0..D).
    # This is much faster than a set.

    def is_valid_add(x, S, used_diffs_mask):
        """Check if adding x to S keeps it Sidon, given used_diffs_mask."""
        # For each s in S, compute d = x - s
        # If d is already in used_diffs_mask, conflict.
        # Also, we need to check that d doesn't conflict with differences
        # among existing elements? No - used_diffs_mask already contains all
        # differences among existing elements.
        # So we just check: for each s in S, is (x - s) in used_diffs_mask?
        for s in S:
            d = x - s
            if d <= 0:
                continue
            if (used_diffs_mask >> d) & 1:
                return False
        return True

    def add_to_mask(used_diffs_mask, x, S):
        """Return new used_diffs_mask after adding x."""
        new_mask = used_diffs_mask
        for s in S:
            d = x - s
            if d > 0:
                new_mask |= (1 << d)
        return new_mask

    def coverage_gain(x, S, covered_mask):
        """Return bitmask of newly covered differences in 1..30."""
        gain = 0
        for s in S:
            d = x - s
            if 1 <= d <= 30:
                gain |= (1 << (d - 1))
        return gain

    def dfs(S, used_diffs_mask, covered_mask, D, depth_limit):
        """
        Depth-first search to find a Sidon set S (extended) with max element <= D,
        covering all differences 1..30, using at most max_size elements.
        Returns the list of elements if found, else None.
        """
        if time.time() > deadline:
            return None

        if (covered_mask & full_mask) == full_mask:
            return list(S)

        if len(S) >= max_size:
            return None

        if len(S) >= depth_limit:
            return None

        # Pruning: can we cover all missing diffs with remaining slots?
        missing = full_mask & ~covered_mask
        if missing == 0:
            return list(S)
        remaining_slots = max_size - len(S)
        # Each new element x can create at most len(S) new differences with existing elements
        # plus differences among new elements. But conservatively, each new element adds
        # at most len(S) + (number of other new elements) differences.
        # Simple bound: remaining_slots * (len(S) + remaining_slots - 1) // 2 + remaining_slots * len(S)
        # Actually each new element x_i creates differences with all previous elements.
        # Total new differences from k new elements = k*len(S) + k*(k-1)//2
        max_new_diffs = remaining_slots * len(S) + remaining_slots * (remaining_slots - 1) // 2
        if bin(missing).count('1') > max_new_diffs:
            return None

        # Generate candidates
        candidates = []
        for x in range(S[-1] + 1, D + 1):
            if not is_valid_add(x, S, used_diffs_mask):
                continue
            gain = coverage_gain(x, S, covered_mask)
            new_covered = covered_mask | gain
            missing_covered = gain & missing
            coverage_count = bin(missing_covered).count('1')
            # Score: prefer more coverage, then smaller x
            candidates.append((-coverage_count, x, gain))

        if not candidates:
            return None

        # Sort by coverage descending, then x ascending
        candidates.sort(key=lambda t: (t[0], t[1]))

        # Limit candidates to control branching
        limit = 50 if D <= 40 else 30
        for _, x, gain in candidates[:limit]:
            new_mask = add_to_mask(used_diffs_mask, x, S)
            new_covered = covered_mask | gain
            S.append(x)
            res = dfs(S, new_mask, new_covered, D, depth_limit)
            S.pop()
            if res is not None:
                return res

        return None

    # Try increasing D from 30 upward
    best_result = None
    for D in range(30, 80):
        if time.time() > deadline - 10:
            break
        # Try with full depth first
        res = dfs([0], 0, 0, D, max_size)
        if res is not None:
            best_result = res
            break

    if best_result is not None:
        return best_result

    # If no exact solution found within time, do a greedy construction
    # that maximizes coverage while maintaining Sidon property
    S = [0]
    used_diffs_mask = 0
    covered_mask = 0
    target = full_mask

    while (covered_mask & full_mask) != full_mask and len(S) < max_size:
        if time.time() > deadline - 2:
            break
        best_x = None
        best_gain = 0
        for x in range(S[-1] + 1, 200):
            if not is_valid_add(x, S, used_diffs_mask):
                continue
            gain = coverage_gain(x, S, covered_mask)
            new_covered = covered_mask | gain
            missing_covered = gain & (full_mask & ~covered_mask)
            cov = bin(missing_covered).count('1')
            if cov > best_gain:
                best_gain = cov
                best_x = x
            elif cov == best_gain and best_x is not None and x < best_x:
                best_x = x

        if best_x is None:
            # No element improves coverage; just add the smallest valid element
            for x in range(S[-1] + 1, 200):
                if is_valid_add(x, S, used_diffs_mask):
                    best_x = x
                    break

        if best_x is None:
            break

        S.append(best_x)
        used_diffs_mask = add_to_mask(used_diffs_mask, best_x, S)
        covered_mask |= coverage_gain(best_x, S[:-1], covered_mask)

    return S
