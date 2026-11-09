import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 145  # Safety margin for 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def check_conflict(x, s):
        """Check if adding x to set s creates a conflict.
        Returns True if conflict exists."""
        if 2 * x in s:
            return True
        for y in s:
            if x + y in s:
                return True
            if y - x in s:
                return True
        return False

    def conflict_count(x, s):
        """Count how many elements in s would conflict with x."""
        count = 0
        if 2 * x in s:
            count += 1
        for y in s:
            if x + y in s:
                count += 1
            if y - x in s:
                count += 1
        return count

    def local_search(s_init, all_values, max_time=30.0):
        """Local search: try to add elements, then try 1-for-1 and 2-for-1 swaps."""
        s = set(s_init)
        c = set(all_values) - s
        deadline = time.time() + max_time
        improved = True

        while improved and time.time() < deadline:
            improved = False

            # Phase 1: Try to add any free element
            for x in list(c):
                if time.time() > deadline:
                    break
                if not check_conflict(x, s):
                    s.add(x)
                    c.discard(x)
                    improved = True

            if time.time() > deadline:
                break

            # Phase 2: Try 1-for-1 swaps to unlock progress
            for x in list(c):
                if time.time() > deadline:
                    break
                blockers = []
                if 2 * x in s:
                    blockers.append(2 * x)
                for y in s:
                    if x + y in s:
                        blockers.append(y)
                    if y - x in s:
                        blockers.append(y)
                if len(blockers) == 1:
                    # Try this swap
                    y_out = blockers[0]
                    s_temp = s - {y_out}
                    s_temp.add(x)
                    # Check if this is still sum-free (it should be by construction, but verify)
                    # Actually, we need to make sure removing y_out and adding x doesn't create new conflicts
                    # among the remaining elements. Since s was sum-free, removing an element keeps it sum-free.
                    # Adding x: we know x only conflicts with y_out in s, so with s\{y_out} it's fine.
                    s = s_temp
                    c.discard(x)
                    c.add(y_out)
                    improved = True
                    break

            if time.time() > deadline:
                break

            # Phase 3: Try 2-for-1 swaps (remove 2 elements, add 1) - only if size would increase
            # Actually this would decrease size. Let's try 1-for-2 (remove 1, add 2)
            for x1 in list(c):
                if time.time() > deadline:
                    break
                # Find elements that x1 blocks
                blockers1 = set()
                if 2 * x1 in s:
                    blockers1.add(2 * x1)
                for y in s:
                    if x1 + y in s:
                        blockers1.add(y)
                    if y - x1 in s:
                        blockers1.add(y)

                for x2 in list(c):
                    if time.time() > deadline:
                        break
                    if x2 == x1:
                        continue
                    # Check if x2 also only conflicts with elements in blockers1
                    blockers2 = set()
                    if 2 * x2 in s:
                        blockers2.add(2 * x2)
                    for y in s:
                        if x2 + y in s:
                            blockers2.add(y)
                        if y - x2 in s:
                            blockers2.add(y)

                    # If both x1 and x2 are only blocked by the same single element
                    if blockers1 and blockers2 and blockers1 == blockers2:
                        y_out = list(blockers1)[0]
                        # Check x1 and x2 don't conflict with each other
                        if x1 + x2 not in s and x1 + x2 != y_out and 2*x1 != x2 and 2*x2 != x1:
                            # Also need x1+x2 not to be in s after removal
                            # Actually x1+x2 could be in s, that would be a problem
                            if x1 + x2 in s and x1 + x2 != y_out:
                                continue
                            s_temp = s - {y_out}
                            s_temp.add(x1)
                            s_temp.add(x2)
                            # Verify sum-free
                            if is_sum_free(list(s_temp)):
                                s = s_temp
                                c.discard(x1)
                                c.discard(x2)
                                c.add(y_out)
                                improved = True
                                break
                if improved:
                    break

        return list(s)

    def greedy_sort(values, mode='degree'):
        """Greedy construction with different sorting strategies."""
        vals = list(values)
        s = set()

        if mode == 'degree':
            # Sort by how many conflicts each element has with the full set
            v_set = set(vals)
            def degree(v):
                cnt = 0
                for u in vals:
                    if u == v:
                        continue
                    if v + u in v_set:
                        cnt += 1
                    if u - v in v_set:
                        cnt += 1
                    if v - u in v_set:
                        cnt += 1
                if 2 * v in v_set:
                    cnt += 1
                return cnt
            vals.sort(key=degree)
        elif mode == 'abs_desc':
            vals.sort(key=abs, reverse=True)
        elif mode == 'abs_asc':
            vals.sort(key=abs)
        elif mode == 'rand':
            random.shuffle(vals)
        elif mode == 'pos_first':
            vals.sort(key=lambda x: (x < 0, abs(x)))
        elif mode == 'neg_first':
            vals.sort(key=lambda x: (x > 0, abs(x)))

        for x in vals:
            if not check_conflict(x, s):
                s.add(x)

        return list(s)

    def solve_instance(values, budget=30.0):
        n = len(values)
        if n <= 2:
            return list(values)

        v_set = set(values)

        # Generate multiple candidate solutions
        candidates = []

        # Structural initializations
        s_odds = set(x for x in values if x % 2 != 0)
        s_pos = set(x for x in values if x > 5000)
        s_neg = set(x for x in values if x < -5000)
        s_large = set(x for x in values if abs(x) > 5000)
        s_mod3 = set(x for x in values if x % 3 != 0)

        candidates.append(local_search(s_odds, values, budget / 3))
        candidates.append(local_search(s_pos, values, budget / 6))
        candidates.append(local_search(s_neg, values, budget / 6))
        candidates.append(local_search(s_large, values, budget / 6))
        candidates.append(local_search(s_mod3, values, budget / 6))

        # Greedy strategies
        for mode in ['degree', 'abs_desc', 'abs_asc', 'pos_first', 'neg_first']:
            g = greedy_sort(values, mode)
            candidates.append(local_search(set(g), values, budget / 10))

        # Random restarts
        for _ in range(8):
            if time.time() > start_time + budget:
                break
            g = greedy_sort(values, 'rand')
            candidates.append(local_search(set(g), values, budget / 20))

        # Pick the best valid candidate
        best = []
        for c in candidates:
            if is_sum_free(c) and len(c) > len(best):
                best = c

        if not best:
            # Fallback: odd numbers
            best = [x for x in values if x % 2 != 0]

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 5:
            vals = instance(size, seed)
            results.append([x for x in vals if x % 2])
            continue

        # Allocate budget proportionally to problem size
        budget = min(remaining - 2, 30 * (size / 240))
        budget = max(budget, 3)

        vals = instance(size, seed)
        res = solve_instance(vals, budget)

        if not is_sum_free(res):
            res = [x for x in vals if x % 2 != 0]

        results.append(res)

    return results
