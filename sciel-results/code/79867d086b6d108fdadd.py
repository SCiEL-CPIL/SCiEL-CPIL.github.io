"""Backtracking search with bitmask diff tracking for minimal-diameter Sidon set."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Try known candidates first
    known_candidates = [
        [0, 1, 3, 7, 12, 21, 25],
        [0, 1, 3, 7, 12, 21, 28],
        [0, 1, 3, 7, 12, 25, 28],
        [0, 2, 5, 10, 18, 23, 28],
        [0, 1, 4, 10, 18, 23, 28],
        [0, 1, 2, 7, 12, 22, 28],
        [0, 1, 3, 8, 12, 22, 26],
        [0, 1, 4, 10, 12, 28, 32, 45, 63],
        [0, 1, 4, 10, 12, 28, 32, 45],
        [0, 1, 4, 10, 12, 28, 32, 45, 47],
        [0, 1, 4, 10, 12, 28, 32, 45, 49],
    ]

    def is_valid(a):
        diffs = set()
        n = len(a)
        for i in range(n):
            for j in range(i + 1, n):
                d = a[j] - a[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return all(d in diffs for d in range(1, 31))

    for c in known_candidates:
        c = sorted(set(c))
        if 1 <= len(c) <= 32 and is_valid(c):
            return c

    # Bitset-based greedy coverage: model as max coverage set packing
    # Each candidate x covers diffs {x - a_i : a_i in current_set, x - a_i in 1..30}
    # Greedily select x maximizing (newly_covered / diameter_increase)
    def greedy_coverage(D):
        """Try to build a Sidon set in {0..D} using greedy coverage."""
        current = [0]
        diff_set = set()
        covered_mask = 0  # bit i set if diff i+1 is covered

        for _ in range(30):  # at most 30 more elements needed
            if time.time() > deadline:
                return None
            if covered_mask == (1 << 30) - 1:
                break

            best_x = None
            best_new_diffs = 0
            best_ratio = 0

            for x in range(1, D + 1):
                if x in current:
                    continue
                # Check Sidon property: x - a must not be in diff_set for any a in current
                ok = True
                new_diffs = []
                for a in current:
                    d = x - a
                    if d <= 0 or d > 30:
                        continue
                    if d in diff_set:
                        ok = False
                        break
                    new_diffs.append(d)
                if not ok:
                    continue

                # Count newly covered required diffs
                new_mask = 0
                for d in new_diffs:
                    if 1 <= d <= 30:
                        new_mask |= (1 << (d - 1))

                newly_covered = bin(new_mask & ~covered_mask).count('1')
                if newly_covered == 0:
                    continue

                # Ratio: newly covered / (x - max(current) + 1) to penalize large diameter
                diameter_increase = x - max(current) if current else x
                if diameter_increase <= 0:
                    diameter_increase = 1
                ratio = newly_covered / diameter_increase

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_x = x
                    best_new_diffs = new_diffs

            if best_x is None:
                return None

            current.append(best_x)
            for d in best_new_diffs:
                diff_set.add(d)
                if 1 <= d <= 30:
                    covered_mask |= (1 << (d - 1))

        # Final check
        if covered_mask == (1 << 30) - 1:
            # Verify Sidon property fully
            all_diffs = set()
            for i in range(len(current)):
                for j in range(i + 1, len(current)):
                    d = current[j] - current[i]
                    if d in all_diffs:
                        return None
                    all_diffs.add(d)
            if all(d in all_diffs for d in range(1, 31)):
                return sorted(current)
        return None

    # Try diameters from lower bound upward using greedy coverage
    for D in range(28, 80):
        if time.time() > deadline:
            break
        result = greedy_coverage(D)
        if result is not None:
            return result

    # Fallback: simple greedy that always finds a valid set
    def simple_greedy():
        a = [0]
        diff_set = set()
        for d in range(1, 31):
            if d in diff_set:
                continue
            # Try to find x such that x - a_i = d for some a_i, and Sidon holds
            found = False
            for x in range(d, 1000):
                ok = True
                new_diffs = []
                for elem in a:
                    dd = x - elem
                    if dd in diff_set:
                        ok = False
                        break
                    new_diffs.append(dd)
                if ok and d in new_diffs:
                    a.append(x)
                    diff_set.update(new_diffs)
                    found = True
                    break
            if not found:
                # Add two elements to force diff d
                y = 2 * max(a) + d + 1
                while True:
                    ok = True
                    new_diffs = []
                    for elem in a:
                        dd = y - elem
                        if dd in diff_set:
                            ok = False
                            break
                        new_diffs.append(dd)
                    if ok:
                        # Check y + d too
                        ok2 = True
                        new_diffs2 = []
                        for elem in a + [y]:
                            dd = (y + d) - elem
                            if dd in diff_set or dd in new_diffs:
                                ok2 = False
                                break
                            new_diffs2.append(dd)
                        if ok2 and d in new_diffs2:
                            a.extend([y, y + d])
                            diff_set.update(new_diffs)
                            diff_set.update(new_diffs2)
                            break
                    y += 1
        return sorted(a)

    return simple_greedy()
# EVOLVE-BLOCK-END
