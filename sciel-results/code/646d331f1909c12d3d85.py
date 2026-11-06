"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        s = set(subset)
        for a in subset:
            for b in subset:
                if a + b in s:
                    return False
        return True

    def solve_instance(values):
        n = len(values)
        # Start with odd numbers as a baseline
        odds = [x for x in values if x % 2 != 0]
        
        # Build conflict graph: edge between a and b if a+b is in the set, or if a+a is in the set
        # For sum-free: we need no a, b, c in S such that a+b=c (a=b allowed)
        # So conflicts: for each pair (a,b) in S, if a+b in S, then we can't have all three.
        # Actually, the constraint is: for any a, b in S, a+b should NOT be in S.
        # This means: for each a in S, and each b in S, a+b not in S.
        # Equivalently: S is sum-free iff for all a, b in S, a+b notin S.
        
        # Greedy approach: try to build a large sum-free set
        # Strategy: sort by some heuristic, greedily add if it doesn't create a conflict
        
        # Better: use local search
        # Start with odds, then try to add evens that don't conflict
        
        S = set(odds)
        
        # Try to add elements one by one, starting with those with fewest conflicts
        remaining = [x for x in values if x not in S]
        
        # Sort remaining by how many conflicts they have with current S
        def conflict_count(x, current_S):
            cnt = 0
            for y in current_S:
                if x + y in current_S:
                    cnt += 1
                if y + x in current_S:
                    cnt += 1
            # Also check x+x
            if x + x in current_S:
                cnt += 1
            return cnt
        
        # Iteratively try to add elements
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Sort remaining by conflict count
            remaining.sort(key=lambda x: conflict_count(x, S))
            for x in remaining:
                if x in S:
                    continue
                # Check if adding x keeps S sum-free
                can_add = True
                for y in S:
                    if x + y in S or y + x in S:
                        can_add = False
                        break
                if x + x in S:
                    can_add = False
                if can_add:
                    S.add(x)
                    improved = True
                    remaining.remove(x)
        
        # Local search: try to swap out elements for better ones
        # Try to remove an element and add two others
        for _ in range(100):
            if time.time() > deadline:
                break
            # Try removing each element and see if we can add more
            best_swap = None
            best_gain = 0
            for x in list(S):
                S_without = S - {x}
                # Try to add elements from remaining
                added = 0
                temp_S = set(S_without)
                for y in remaining:
                    if y in temp_S:
                        continue
                    can_add = True
                    for z in temp_S:
                        if y + z in temp_S or z + y in temp_S:
                            can_add = False
                            break
                    if y + y in temp_S:
                        can_add = False
                    if can_add:
                        temp_S.add(y)
                        added += 1
                if added - 1 > best_gain:
                    best_gain = added - 1
                    best_swap = (x, temp_S)
            if best_swap is not None:
                x, new_S = best_swap
                S = new_S
                remaining = [y for y in values if y not in S]
        
        return sorted(S)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Return best effort
            values = instance(size, seed)
            odds = [x for x in values if x % 2 != 0]
            results.append(odds)
            continue
        values = instance(size, seed)
        subset = solve_instance(values)
        # Verify sum-free
        if not is_sum_free(subset):
            # Fallback to odds
            subset = [x for x in values if x % 2 != 0]
        results.append(subset)

    return results
# EVOLVE-BLOCK-END
