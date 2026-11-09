import time
import random
import numpy as np

def construct():
    t_start = time.time()
    DEADLINE = t_start + 78.0
    N = 4096

    def _compute_sums(selected):
        n = len(selected)
        seen = set()
        for a in range(n):
            sa = selected[a]
            for b in range(a, n):
                sb = selected[b]
                sa_b = sa + sb
                for c in range(b, n):
                    s = sa_b + selected[c]
                    if s in seen:
                        return None
                    seen.add(s)
        return seen

    def _incremental_check_add(selected, x, existing_sums):
        n = len(selected)
        new_sums = set()
        # (x,x,x)
        s = 3 * x
        if s in existing_sums or s in new_sums:
            return (False, None)
        new_sums.add(s)
        # (a,x,x)
        for a in selected:
            s = a + 2 * x
            if s in existing_sums or s in new_sums:
                return (False, None)
            new_sums.add(s)
        # (a,b,x) with a<=b
        for i in range(n):
            sa = selected[i]
            for j in range(i, n):
                sb = selected[j]
                s = sa + sb + x
                if s in existing_sums or s in new_sums:
                    return (False, None)
                new_sums.add(s)
        return (True, existing_sums | new_sums)

    def greedy_smallest(start_list, limit, deadline, max_size=80):
        base = sorted(start_list)
        if not base:
            base = [1]
        sums = _compute_sums(base)
        if sums is None:
            return base
        while len(base) < max_size:
            if time.time() > deadline:
                break
            next_min = max(base) + 1
            found = False
            for x in range(next_min, limit + 1):
                ok, new_sums = _incremental_check_add(base, x, sums)
                if ok:
                    base.append(x)
                    sums = new_sums
                    found = True
                    break
            if not found:
                break
        return base

    def _valid_set(check_set, limit):
        """Return sorted list if valid, else None."""
        if len(check_set) == 0:
            return []
        if len(check_set) != len(set(check_set)):
            return None
        arr = sorted(check_set)
        if arr[-1] > limit:
            return None
        return arr

    def _remove_and_recheck(selected, sums, remove_idx, limit):
        """Remove element at remove_idx and try to add two smaller replacements.
        Returns (improved_list, new_sums) or None."""
        n = len(selected)
        candidate = selected[:remove_idx] + selected[remove_idx+1:]
        new_sums = _compute_sums(candidate)
        if new_sums is None:
            return None

        # Try to add two elements to get back to n+1
        best_local = list(candidate)
        local_sums = set(new_sums)
        # We want to add elements that keep the set valid
        # Try from smallest available
        added_count = 0
        # Try to add the smallest possible first, then next smallest
        for _ in range(2):
            if time.time() > deadline:
                return None
            found = False
            start = min(candidate) if candidate else 1
            end = max(candidate)
            # Try values from 1 up to end
            for x in range(1, end + 1):
                if x in local_sums:
                    continue
                ok, updated = _incremental_check_add(best_local, x, local_sums)
                if ok:
                    best_local.append(x)
                    local_sums = updated
                    added_count += 1
                    found = True
                    break
            if not found:
                return None

        if len(best_local) > len(selected):
            return (best_local, local_sums)
        return None

    def local_search(base, limit, deadline, max_iter=20):
        base = sorted(base)
        if not base:
            return base
        sums = _compute_sums(base)
        if sums is None:
            return base

        best = list(base)
        best_size = len(base)

        for it in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for idx in range(len(base)):
                if time.time() > deadline:
                    break
                result = _remove_and_recheck(base, sums, idx, limit)
                if result is not None:
                    new_base, new_sums = result
                    if len(new_base) > best_size:
                        best = new_base
                        best_size = len(best)
                        improved = True
                        break
            if not improved:
                break
        return best

    def randomized_greedy(order, deadline):
        vals = []
        existing_sums = set()
        for x in order:
            if time.time() > deadline:
                break
            k = len(vals)
            conflict = False
            new_sums = []
            s = 3 * x
            if s in existing_sums:
                conflict = True
            else:
                new_sums.append(s)
            if not conflict:
                for v in vals:
                    s = 2 * x + v
                    if s in existing_sums or s in new_sums:
                        conflict = True
                        break
                    new_sums.append(s)
            if not conflict:
                for i in range(k):
                    v1 = vals[i]
                    for j in range(i, k):
                        v2 = vals[j]
                        s = x + v1 + v2
                        if s in existing_sums or s in new_sums:
                            conflict = True
                            break
                        new_sums.append(s)
                    if conflict:
                        break
            if not conflict:
                vals.append(x)
                existing_sums.update(new_sums)
        return vals

    best = []

    # Phase 1: Deterministic greedy from 1
    order1 = list(range(1, N + 1))
    result = randomized_greedy(order1, DEADLINE)
    if len(result) > len(best):
        best = result

    # Phase 2: Incremental greedy (smallest valid next)
    if time.time() < DEADLINE - 2:
        result2 = greedy_smallest([1], N, DEADLINE)
        if len(result2) > len(best):
            best = result2

    # Phase 3: Randomized restarts
    rng = random.Random(12345)
    for attempt in range(500):
        if time.time() > DEADLINE - 5:
            break

        strategy = attempt % 6
        if strategy == 0:
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif strategy == 1:
            offset = rng.randint(1, 150)
            order = list(range(1 + offset, N + 1)) + list(range(1, 1 + offset))
        elif strategy == 2:
            small = list(range(1, 100))
            large = list(range(100, N + 1))
            rng.shuffle(small)
            rng.shuffle(large)
            order = small + large
        elif strategy == 3:
            center = rng.randint(1, N)
            order = [center]
            for d in range(1, N):
                if center - d >= 1:
                    order.append(center - d)
                if center + d <= N:
                    order.append(center + d)
        elif strategy == 4:
            # Start with random small valid seed, then extend
            k = rng.randint(3, 6)
            initial = sorted(rng.sample(range(1, N + 1), k))
            if _compute_sums(initial) is not None:
                result = greedy_smallest(initial, N, DEADLINE)
                if len(result) > len(best):
                    best = result
                continue
        else:
            # Geometric-ish spacing
            ratio = rng.uniform(1.5, 2.5)
            order = []
            cur = rng.randint(1, 50)
            while cur <= N:
                order.append(cur)
                cur = int(cur * ratio) + rng.randint(0, 5)
            order = sorted(set(order))
            if not order:
                order = [1]

        result = randomized_greedy(order, DEADLINE)
        if len(result) > len(best):
            best = result
            if len(best) >= 60:
                break

    # Phase 4: Local search polish
    if time.time() < DEADLINE - 10 and len(best) > 0:
        improved = local_search(best, N, DEADLINE, max_iter=15)
        if len(improved) > len(best):
            best = improved

    # Final validation
    if best:
        final_check = _compute_sums(best)
        if final_check is None:
            # Fallback to a known small valid set
            best = [1, 3, 7, 12]
            # Try to extend
            best = greedy_smallest(best, N, DEADLINE)

    return best
