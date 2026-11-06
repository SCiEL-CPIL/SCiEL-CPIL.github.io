"""Sum-free subset maximizer with greedy + local search."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free."""
        S_set = set(S)
        for a in S:
            for b in S:
                if a + b in S_set:
                    return False
        return True

    def greedy_add(S, candidates, order_key=None):
        """Greedily add elements from candidates to S maintaining sum-free."""
        S_set = set(S)
        result = list(S)
        # Determine order
        if order_key is not None:
            candidates = sorted(candidates, key=order_key)
        else:
            candidates = list(candidates)
        
        for x in candidates:
            if x in S_set:
                continue
            # Check if adding x keeps sum-free
            # Need: for all a in S_set, a + x not in S_set, and x + x not in S_set
            # And for all b in S_set, x + b not in S_set (same as above)
            ok = True
            if x + x in S_set:
                ok = False
            else:
                for a in S_set:
                    if a + x in S_set:
                        ok = False
                        break
            if ok:
                result.append(x)
                S_set.add(x)
        return result

    def local_search(S, full_set, max_iters=2000):
        """Local search: try adding excluded elements, and 1-for-1 swaps."""
        S_set = set(S)
        excluded = [x for x in full_set if x not in S_set]
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            # Try adding each excluded element
            for i, x in enumerate(excluded):
                # Check if x can be added
                ok = True
                if x + x in S_set:
                    ok = False
                else:
                    for a in S_set:
                        if a + x in S_set:
                            ok = False
                            break
                if ok:
                    S_set.add(x)
                    excluded.pop(i)
                    improved = True
                    break
            if improved:
                continue
            # Try 1-for-1 swaps: remove s from S, add x from excluded
            # This is expensive, so limit
            for s in list(S_set):
                S_set.remove(s)
                for x in excluded:
                    if x == s:
                        continue
                    ok = True
                    if x + x in S_set:
                        ok = False
                    else:
                        for a in S_set:
                            if a + x in S_set:
                                ok = False
                                break
                    if ok:
                        S_set.add(x)
                        excluded.remove(x)
                        improved = True
                        break
                if improved:
                    break
                else:
                    S_set.add(s)
            if not improved and iters > 100:
                break
        return list(S_set)

    def solve_one(full_set, time_budget):
        """Solve for one instance, returning best sum-free subset."""
        best = []
        best_size = 0
        n = len(full_set)
        
        # Strategy 1: Odd numbers
        odd_set = [x for x in full_set if x % 2 != 0]
        if len(odd_set) > best_size:
            best = odd_set
            best_size = len(odd_set)
        
        # Strategy 2: Greedy with different orderings
        orderings = [
            None,  # original order
            lambda x: abs(x),  # by magnitude
            lambda x: -abs(x),  # by negative magnitude
            lambda x: x,  # ascending
            lambda x: -x,  # descending
        ]
        
        for order_key in orderings:
            if time.time() > time_budget:
                break
            # Start empty, greedy add
            S = greedy_add([], full_set, order_key)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        # Strategy 3: Start from odd set, try to add evens
        if time.time() <= time_budget:
            S = greedy_add(odd_set, [x for x in full_set if x % 2 == 0], None)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        # Strategy 4: Local search from best
        if time.time() <= time_budget and best_size > 0:
            S = local_search(best, full_set, max_iters=500)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        # Strategy 5: Randomized greedy with multiple restarts
        if time.time() <= time_budget:
            rng = Random(42)
            for restart in range(5):
                if time.time() > time_budget:
                    break
                shuffled = full_set[:]
                rng.shuffle(shuffled)
                S = greedy_add([], shuffled, None)
                if len(S) > best_size:
                    best = S
                    best_size = len(S)
        
        return best

    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for size, seed in instances:
        full_set = instance(size, seed)
        # Allocate time: more time for larger instances
        remaining = deadline - time.time()
        if remaining < 5:
            # Not enough time, just use odd numbers
            results.append([x for x in full_set if x % 2 != 0])
            continue
        # Give proportional time
        time_budget = time.time() + remaining * 0.3
        if time_budget > deadline - 10:
            time_budget = deadline - 10
        best = solve_one(full_set, time_budget)
        results.append(best)
    
    # Final safety check
    if time.time() > deadline - 5:
        # Return what we have, ensure validity
        for i, S in enumerate(results):
            if not is_sum_free(S):
                # Fallback to odd numbers
                full_set = instance(instances[i][0], instances[i][1])
                results[i] = [x for x in full_set if x % 2 != 0]
    
    return results
# EVOLVE-BLOCK-END
