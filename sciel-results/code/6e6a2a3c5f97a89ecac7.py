"""Rewritten program for large sum-free subsets of three fixed sets."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        """Check if set S is sum-free: no a+b=c with a,b,c in S (a=b allowed)."""
        S_set = set(S)
        for a in S_set:
            for b in S_set:
                if a + b in S_set:
                    return False
        return True

    def can_add(S, x):
        """Check if x can be added to S without violating sum-free."""
        # Violations: a+x in S, x+a in S (same), or a+a=x for a in S
        for a in S:
            if a + x in S:
                return False
            if a + a == x:
                return False
        return True

    def greedy_build(values, order_key=None, seed=0):
        """Greedy: add elements in given order if they don't create violation."""
        if order_key is None:
            order = list(values)
        else:
            order = sorted(values, key=order_key)
        
        S = set()
        for x in order:
            if can_add(S, x):
                S.add(x)
        return S

    def local_search(S, values, time_limit=None):
        """Improve S using 1-add, 1-remove, and 2-swap moves."""
        S_set = set(S)
        all_vals = set(values)
        excluded = all_vals - S_set
        
        def count_conflicts(x, S):
            """Count how many pairs (a,b) in S violate with x."""
            cnt = 0
            for a in S:
                if a + x in S:
                    cnt += 1
                if a + a == x:
                    cnt += 1
            return cnt

        for iteration in range(100000):
            if time_limit is not None and time.time() > time_limit:
                break
            
            # Try 1-add: find an excluded element that can be added
            added = False
            for x in list(excluded):
                if can_add(S_set, x):
                    S_set.add(x)
                    excluded.discard(x)
                    added = True
                    break
            
            if added:
                continue
            
            # Try 1-remove: remove element with most conflicts, then re-add
            best_remove = None
            best_conflicts = 0
            for a in S_set:
                # Count conflicts that a is involved in
                conflicts = 0
                for b in S_set:
                    if a + b in S_set:
                        conflicts += 1
                    if a + a in S_set:
                        conflicts += 1
                if conflicts > best_conflicts:
                    best_conflicts = conflicts
                    best_remove = a
            
            if best_remove is not None and best_conflicts > 0:
                # Try removing best_remove and see if we can add more
                S_test = S_set - {best_remove}
                excluded_test = excluded | {best_remove}
                gain = 0
                for x in list(excluded_test):
                    if can_add(S_test, x):
                        gain += 1
                if gain > 1:  # net positive
                    S_set = S_test
                    excluded = excluded_test
                    # Re-add what we can
                    for x in list(excluded):
                        if can_add(S_set, x):
                            S_set.add(x)
                            excluded.discard(x)
                    continue
            
            # Try 2-swap: remove one, add another
            found_swap = False
            for a in list(S_set):
                S_test = S_set - {a}
                for x in excluded:
                    if can_add(S_test, x):
                        S_set = S_test | {x}
                        excluded = (excluded - {x}) | {a}
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if not found_swap:
                break
        
        return S_set

    def solve_instance(values, time_budget):
        """Find a large sum-free subset of values within time_budget seconds."""
        n = len(values)
        best = set()
        best_size = 0
        
        # Strategy 1: Odd numbers
        odd = {x for x in values if x % 2 != 0}
        if len(odd) > best_size:
            best = odd
            best_size = len(odd)
        
        # Strategy 2: Upper half (positive large numbers)
        # Numbers > max/2 can't sum to another number in the set if all are positive
        # But we have negatives too, so this is trickier
        # Try: numbers whose absolute value is large
        sorted_by_abs = sorted(values, key=abs, reverse=True)
        upper_half = set(sorted_by_abs[:n//2 + 1])
        if len(upper_half) > best_size:
            # Verify it's sum-free
            if is_sum_free(upper_half):
                best = upper_half
                best_size = len(upper_half)
        
        # Strategy 3: Multiple greedy approaches
        greedy_seeds = [0, 1, 42, 123, 999, 7, 13, 21]
        for seed in greedy_seeds:
            if time.time() > start_time + time_budget - 5:
                break
            S = greedy_build(values, seed=seed)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        # Greedy with different orderings
        orderings = [
            lambda x: x,  # natural order
            lambda x: -x,  # reverse
            lambda x: abs(x),  # by abs value
            lambda x: -abs(x),  # by abs value descending
            lambda x: x if x > 0 else -x,  # positives first by magnitude
        ]
        for ok in orderings:
            if time.time() > start_time + time_budget - 5:
                break
            S = greedy_build(values, order_key=ok)
            if len(S) > best_size:
                best = S
                best_size = len(S)
        
        # Local search from best
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 10:
            improved = local_search(best, values, time_limit=start_time + time_budget - 5)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
        
        # Multiple local search restarts with different initial solutions
        remaining_time = time_budget - (time.time() - start_time)
        if remaining_time > 20:
            # Try local search from different starting points
            for seed in [0, 42, 123]:
                if time.time() > start_time + time_budget - 10:
                    break
                S_init = greedy_build(values, seed=seed)
                S_ls = local_search(S_init, values, time_limit=start_time + time_budget - 10)
                if len(S_ls) > best_size:
                    best = S_ls
                    best_size = len(S_ls)
        
        return best

    # Allocate time budget across instances
    # Instance sizes: 60, 120, 240
    # Give more time to larger instances
    time_budgets = [20.0, 35.0, 75.0]  # total ~130s, leaving margin
    
    results = []
    for i, (size, seed) in enumerate(((60, 801), (120, 802), (240, 803))):
        if time.time() > DEADLINE - 3:
            # Fallback: odd numbers
            vals = instance(size, seed)
            results.append(sorted([x for x in vals if x % 2 != 0]))
            continue
        
        vals = instance(size, seed)
        S = solve_instance(vals, time_budgets[i])
        results.append(sorted(S))
    
    return results
# EVOLVE-BLOCK-END
