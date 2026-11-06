import time
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115

    # Known solution as fallback
    KNOWN = [0, 1, 4, 10, 12, 28, 32, 45, 63]

    def is_valid(arr):
        if len(arr) > 32 or 0 not in arr or max(arr) > 10**6:
            return False
        diffs = set()
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                d = arr[j] - arr[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    def solve_min_diameter(max_diameter, max_time, initial_set=None):
        """
        Find a Sidon set with diameter <= max_diameter using optimized backtracking.
        Returns the set if found, None otherwise.
        """
        best = None
        FULL_MASK = (1 << 30) - 1

        if initial_set is None:
            initial_set = [0]
        
        # Pre-calculate diffs for initial set
        used_diffs = set()
        coverage_mask = 0
        for i in range(len(initial_set)):
            for j in range(i + 1, len(initial_set)):
                d = initial_set[j] - initial_set[i]
                used_diffs.add(d)
                if 1 <= d <= 30:
                    coverage_mask |= (1 << (d - 1))

        next_min = initial_set[-1] + 1

        def backtrack(selected, used_diffs, coverage_mask, next_min):
            nonlocal best
            if time.time() > max_time:
                return
            if coverage_mask == FULL_MASK:
                if best is None or max(selected) < max(best):
                    best = selected[:]
                return

            if len(selected) >= 32:
                return

            if next_min > max_diameter:
                return

            # Strong pruning: count remaining differences to cover
            remaining_mask = FULL_MASK ^ coverage_mask
            remaining_count = remaining_mask.bit_count()
            if remaining_count > 0:
                max_pick = min(32 - len(selected), max_diameter - next_min + 1)
                possible = False
                for k in range(1, max_pick + 1):
                    max_new_diffs = len(selected) * k + k * (k - 1) // 2
                    if max_new_diffs >= remaining_count:
                        possible = True
                        break
                if not possible:
                    return

            # Try candidates in order of coverage potential
            candidates = []
            for x in range(next_min, max_diameter + 1):
                new_diffs = []
                valid = True
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)

                if not valid:
                    continue

                # Compute coverage gain
                gain = 0
                for d in new_diffs:
                    if 1 <= d <= 30:
                        bit = 1 << (d - 1)
                        if not (coverage_mask & bit):
                            gain += 1

                candidates.append((gain, x, new_diffs))

            # Sort by coverage gain descending, then by x ascending
            candidates.sort(key=lambda t: (-t[0], t[1]))

            for gain, x, new_diffs in candidates:
                if time.time() > max_time:
                    return

                new_coverage = coverage_mask
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_coverage |= (1 << (d - 1))

                new_selected = selected + [x]
                new_used = used_diffs | set(new_diffs)
                backtrack(new_selected, new_used, new_coverage, x + 1)

                if best is not None:
                    return

        backtrack(initial_set, used_diffs, coverage_mask, next_min)
        return best

    # Try increasing diameters starting from a lower bound
    # Force 1 in the set for faster convergence
    for D in range(50, 121):
        if time.time() > deadline - 8:
            break
        r = solve_min_diameter(D, deadline - 5, [0, 1])
        if r is not None and is_valid(r):
            return r

    # Try larger diameters with more time, still forcing 1
    for D in range(120, 251):
        if time.time() > deadline - 3:
            break
        r = solve_min_diameter(D, deadline - 2, [0, 1])
        if r is not None and is_valid(r):
            return r

    # Greedy fallback
    def greedy():
        selected = [0, 1]
        used = {1}
        for target in range(2, 31):
            if target in used:
                continue
            best_x = None
            best_cover = -1
            for x in range(2, 5000):
                if x in selected:
                    continue
                conflict = False
                new_diffs = []
                for s in selected:
                    d = x - s
                    if d <= 0:
                        continue
                    if d in used:
                        conflict = True
                        break
                    new_diffs.append(d)
                if conflict:
                    continue
                cover = sum(1 for d in new_diffs if 1 <= d <= 30 and d not in used)
                if target in new_diffs:
                    best_x = x
                    break
                elif cover > best_cover:
                    best_x = x
                    best_cover = cover
            if best_x is not None:
                selected.append(best_x)
                for s in selected[:-1]:
                    d = best_x - s
                    if d > 0:
                        used.add(d)
        if is_valid(selected):
            return sorted(selected)
        return None

    result = greedy()
    if result is not None:
        return result

    r = solve_min_diameter(500, deadline - 1, [0, 1])
    if r is not None and is_valid(r):
        return r

    return KNOWN

# EVOLVE-BLOCK-END
