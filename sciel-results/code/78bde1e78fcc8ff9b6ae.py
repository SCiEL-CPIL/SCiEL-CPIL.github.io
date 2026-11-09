"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # safety margin before 90s kill
    N = 4096
    MAX_SIZE = 80

    def build_greedy(start_from=1, max_size=MAX_SIZE):
        """
        Greedy: add numbers in increasing order, maintaining set of triple sums.
        For each candidate x, check that no new triple sum collides with existing ones.
        Triple sums involving x are:
          - x+x+x = 3x
          - x+x+y for each y in selected (y != x)
          - x+y+z for each pair (y,z) from selected with y<=z
        """
        selected = []
        triple_sums = set()

        for x in range(start_from, N + 1):
            if len(selected) >= max_size:
                break
            if time.time() > DEADLINE - 5:
                break

            # Check if x can be added
            # New triple sums to verify:
            # 1. (x, x, x) -> 3x
            # 2. (x, x, y) for each y in selected -> 2x + y
            # 3. (x, y, z) for each pair (y, z) from selected with y <= z -> x + y + z

            # First check 3x
            if 3 * x in triple_sums:
                continue

            # Check 2x + y for each y in selected
            ok = True
            for y in selected:
                s = 2 * x + y
                if s in triple_sums:
                    ok = False
                    break
            if not ok:
                continue

            # Check x + y + z for each pair (y, z) from selected, y <= z
            n_sel = len(selected)
            for i in range(n_sel):
                y = selected[i]
                for j in range(i, n_sel):
                    z = selected[j]
                    s = x + y + z
                    if s in triple_sums:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            # All checks passed, add x
            selected.append(x)
            triple_sums.add(3 * x)
            for y in selected[:-1]:  # all previously selected
                triple_sums.add(2 * x + y)
            n_sel = len(selected) - 1  # exclude x itself from pair loop
            for i in range(n_sel):
                y = selected[i]
                for j in range(i, n_sel):
                    z = selected[j]
                    triple_sums.add(x + y + z)

        return selected

    # Primary greedy pass
    result = build_greedy(start_from=1, max_size=MAX_SIZE)

    # If we didn't reach max size, try a few alternative strategies
    if len(result) < MAX_SIZE and time.time() < DEADLINE - 10:
        # Try starting from different offsets
        for offset in [2, 3, 5, 7, 11]:
            if time.time() > DEADLINE - 5:
                break
            alt = build_greedy(start_from=offset, max_size=MAX_SIZE)
            if len(alt) > len(result):
                result = alt
            if len(result) >= MAX_SIZE:
                break

    # Verify validity of the result
    def is_valid(arr):
        if len(arr) < 1:
            return False
        if len(set(arr)) != len(arr):
            return False
        if any(x < 1 or x > N for x in arr):
            return False
        from itertools import combinations_with_replacement
        sums = [sum(t) for t in combinations_with_replacement(arr, 3)]
        return len(sums) == len(set(sums))

    if not is_valid(result):
        # Fallback: very conservative greedy
        result = []
        triple_sums = set()
        for x in range(1, N + 1):
            if len(result) >= MAX_SIZE:
                break
            if time.time() > DEADLINE - 2:
                break
            # Full check
            ok = True
            test_arr = result + [x]
            from itertools import combinations_with_replacement
            new_sums = [sum(t) for t in combinations_with_replacement(test_arr, 3)]
            for s in new_sums:
                if s in triple_sums:
                    ok = False
                    break
            if ok:
                result.append(x)
                for s in new_sums:
                    triple_sums.add(s)

    return result
# EVOLVE-BLOCK-END
