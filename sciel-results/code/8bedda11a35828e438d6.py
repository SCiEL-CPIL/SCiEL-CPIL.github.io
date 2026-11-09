"""Optimized sum-free subset finder using multiple greedy strategies + local search."""
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin (150s kill)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c for a,b,c in S (a=b allowed)."""
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def greedy_by_order(S, order):
        """Greedy: add elements in given order if they don't create a violation."""
        selected = []
        selected_set = set()
        for x in order:
            if x in selected_set:
                continue
            # Check: for all a,b in selected, a+b != x and x+a != b (i.e., x not causing violation)
            # Violation: exists a,b in selected such that a+b=x OR x+a=b OR x+b=a
            # Since we only add x, new violations are:
            #   1) a + b = x for some a,b in selected
            #   2) x + a = b for some a,b in selected  => b - a = x
            #   3) a + x = b for some a,b in selected  => b - a = x (same as 2)
            # Actually simpler: x causes violation if:
            #   - exists a,b in selected: a+b == x
            #   - exists a in selected: x+a in selected (i.e., exists b in selected: b-a==x)
            #   - exists a in selected: a+x in selected (same condition)
            ok = True
            for a in selected:
                if x - a in selected_set:
                    # a + (x-a) = x, violation
                    ok = False
                    break
                if x + a in selected_set:
                    # x + a = (x+a), and x+a is in selected, violation
                    ok = False
                    break
            if ok:
                selected.append(x)
                selected_set.add(x)
        return selected

    def local_search(S, max_iters=5000):
        """Try to improve sum-free set by removing elements that cause most conflicts, then re-adding others."""
        S = list(S)
        Sset = set(S)
        best = S[:]
        best_size = len(S)

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            # Find element causing most violations (conflicts)
            # A violation is a triple (a,b,c) with a+b=c where a,b,c in Sset
            # For each element, count how many violations it participates in
            conflict_count = {x: 0 for x in Sset}
            for a in Sset:
                for b in Sset:
                    c = a + b
                    if c in Sset:
                        conflict_count[a] = conflict_count.get(a, 0) + 1
                        conflict_count[b] = conflict_count.get(b, 0) + 1
                        conflict_count[c] = conflict_count.get(c, 0) + 1

            if not any(v > 0 for v in conflict_count.values()):
                break  # Already sum-free

            # Remove the element with highest conflict count
            worst = max(Sset, key=lambda x: conflict_count.get(x, 0))
            S.remove(worst)
            Sset.discard(worst)

            # Try to re-add elements from original set that were removed
            # Actually, let's try a different approach: after removing, try adding back any element
            # from the full instance that fits
            # For simplicity, just try to add elements from a candidate pool
            # We'll skip re-adding for now and just do removal

            if len(S) > best_size:
                best = S[:]
                best_size = len(S)

        return best

    def local_search_v2(S, full_set, max_iters=20000):
        """Better local search: remove one element, then try to add back best candidate."""
        S = list(S)
        Sset = set(S)
        candidates = list(full_set)  # all elements from the instance

        def can_add(Sset, x):
            for a in Sset:
                if x - a in Sset:
                    return False
                if x + a in Sset:
                    return False
            return True

        for iteration in range(max_iters):
            if time.time() > deadline:
                break

            # Count conflicts
            conflict_count = {}
            for a in Sset:
                for b in Sset:
                    c = a + b
                    if c in Sset:
                        conflict_count[a] = conflict_count.get(a, 0) + 1
                        conflict_count[b] = conflict_count.get(b, 0) + 1
                        conflict_count[c] = conflict_count.get(c, 0) + 1

            if not conflict_count:
                # No conflicts, try to add more
                added = False
                for x in candidates:
                    if x not in Sset and can_add(Sset, x):
                        S.append(x)
                        Sset.add(x)
                        added = True
                        break
                if not added:
                    break
            else:
                # Remove worst element
                worst = max(Sset, key=lambda x: conflict_count.get(x, 0))
                S.remove(worst)
                Sset.discard(worst)

                # Try to add back some element
                for x in candidates:
                    if x not in Sset and can_add(Sset, x):
                        S.append(x)
                        Sset.add(x)
                        break

        return S

    def solve_instance(S, time_budget=25):
        """Solve one instance with multiple strategies."""
        S = list(S)
        n = len(S)

        # Strategy 1: Greedy by value (ascending)
        order1 = sorted(S)
        sol1 = greedy_by_order(S, order1)

        # Strategy 2: Greedy by absolute value (ascending)
        order2 = sorted(S, key=lambda x: abs(x))
        sol2 = greedy_by_order(S, order2)

        # Strategy 3: Greedy by value (descending)
        order3 = sorted(S, reverse=True)
        sol3 = greedy_by_order(S, order3)

        # Strategy 4: Greedy by absolute value (descending)
        order4 = sorted(S, key=lambda x: -abs(x))
        sol4 = greedy_by_order(S, order4)

        # Strategy 5: Random orderings
        rng = random.Random(42)
        best_sol = max([sol1, sol2, sol3, sol4], key=len)

        t_start = time.time()
        while time.time() - t_start < time_budget:
            order5 = S[:]
            rng.shuffle(order5)
            sol5 = greedy_by_order(S, order5)
            if len(sol5) > len(best_sol):
                best_sol = sol5

        # Local search on best solution
        best_sol = local_search_v2(best_sol, S, max_iters=50000)

        return best_sol

    # Solve each instance
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]

    for size, seed in instances:
        S = instance(size, seed)
        sol = solve_instance(S, time_budget=40)
        results.append(sol)

    # Verify sum-free property
    for sol in results:
        if not is_sum_free(sol):
            # Fallback to odd numbers if something went wrong
            sol = [x for x in instance(*instances[results.index(sol)]) if x % 2]
            results[results.index(sol)] = sol

    return results
