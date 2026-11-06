"""Sum-free subset maximization via greedy + local search."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 130  # safety margin before 150s kill

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
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_build(elements, order):
        """Greedy: add elements in given order if it keeps the set sum-free."""
        S = []
        S_set = set()
        for x in order:
            # Check: for all a in S, a + x not in S_set, and x + a not in S_set
            # Also: x + x not in S_set (i.e., 2x not in S)
            # And: for all a in S, a + x not in S_set (already covered)
            # Actually need: for all a in S, a+x not in S_set AND x+x not in S_set
            # Wait: a+b=c means we need no pair (a,b) in S such that a+b in S.
            # When adding x: check that for every a in S, a+x is not in S_set (new c)
            # and x+x is not in S_set. Also x+a is same as a+x.
            ok = True
            if 2 * x in S_set:
                ok = False
            if ok:
                for a in S:
                    if a + x in S_set:
                        ok = False
                        break
            if ok:
                S.append(x)
                S_set.add(x)
        return S

    def local_search(S, elements, max_iters=5000, timeout=10.0):
        """Try to improve S by swapping out one element and in another."""
        S_set = set(S)
        S_list = list(S)
        available = [x for x in elements if x not in S_set]
        best_size = len(S_list)
        no_improve_count = 0
        t0 = time.time()

        while max_iters > 0 and no_improve_count < 200:
            max_iters -= 1
            if time.time() - t0 > timeout:
                break
            # Try random swap: remove r from S, add a from available
            r_idx = random.randrange(len(S_list))
            r = S_list[r_idx]
            a = random.choice(available)
            if a == r:
                continue
            # Tentative: remove r, add a
            S_set.remove(r)
            S_list.pop(r_idx)
            # Check if adding a keeps sum-free
            ok = True
            if 2 * a in S_set:
                ok = False
            if ok:
                for b in S_list:
                    if b + a in S_set:
                        ok = False
                        break
            if not ok:
                # Revert
                S_list.append(r)
                S_set.add(r)
                continue
            # Add a
            S_list.append(a)
            S_set.add(a)
            best_size = len(S_list)
            no_improve_count = 0
            # Update available
            if a in available:
                available.remove(a)
            available.append(r)
        return S_list

    def solve_instance(elements, time_budget=30.0):
        """Solve one instance: try multiple greedy orderings, then local search."""
        n = len(elements)
        best = []
        best_size = 0

        # Multiple greedy orderings
        orderings = []
        # Natural order
        orderings.append(list(elements))
        # Reverse
        orderings.append(list(reversed(elements)))
        # By absolute value
        orderings.append(sorted(elements, key=abs))
        orderings.append(sorted(elements, key=abs, reverse=True))
        # By value
        orderings.append(sorted(elements))
        # Random shuffles
        rng = random.Random(42)
        for _ in range(5):
            o = list(elements)
            rng.shuffle(o)
            orderings.append(o)

        for order in orderings:
            if time.time() - start_time > DEADLINE:
                break
            S = greedy_build(elements, order)
            if len(S) > best_size:
                best_size = len(S)
                best = list(S)

        # Local search from best greedy result
        if time.time() - start_time < DEADLINE - 5:
            improved = local_search(best, elements, max_iters=20000, timeout=min(time_budget, DEADLINE - time.time() - 5))
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

        # Multiple local search restarts from different greedy solutions
        for order in orderings[:3]:
            if time.time() - start_time > DEADLINE - 5:
                break
            S = greedy_build(elements, order)
            improved = local_search(S, elements, max_iters=10000, timeout=min(10.0, DEADLINE - time.time() - 5))
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)

        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() - start_time > DEADLINE:
            results.append([])
            continue
        elems = instance(size, seed)
        # Allocate more time to larger instances
        budget = 40.0 if size <= 60 else (30.0 if size <= 120 else 25.0)
        sol = solve_instance(elems, time_budget=budget)
        # Verify sum-free
        if not is_sum_free(sol):
            # Fallback to odd numbers
            sol = [x for x in elems if x % 2 != 0]
        results.append(sol)

    return results
# EVOLVE-BLOCK-END
