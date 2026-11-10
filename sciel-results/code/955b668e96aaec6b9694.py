import random
import time
from fractions import Fraction
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin well under the 90s kill limit

    # Precompute reciprocals as exact Fractions
    recip = {i: Fraction(1, i) for i in range(1, 25)}

    # Cache for validity checks: frozenset -> bool
    valid_cache = {}

    def is_valid(S):
        """
        Returns True iff S is reciprocal-subsum-free:
        for every x in S, 1/x is NOT the sum of reciprocals of any subset of S \ {x}.
        Uses meet-in-the-middle with pruning.
        Optimization: check smallest targets first (they have the most candidates
        to sum to them, so violations are more likely found early).
        """
        key = frozenset(S)
        if key in valid_cache:
            return valid_cache[key]
        
        S_list = sorted(S)
        n = len(S_list)
        if n < 2:
            valid_cache[key] = True
            return True

        # Check elements in order of increasing value (smallest x has largest 1/x,
        # making it hardest to represent as a sum, so check largest 1/x first = smallest x)
        # Actually, smallest x => largest 1/x => harder to hit with sums of smaller reciprocals
        # Let's check in order of decreasing 1/x (i.e., increasing x) to find violations early
        for x in S_list:
            target = recip[x]
            others = [y for y in S_list if y != x]
            if not others:
                continue

            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]

            # Build right subset sums with pruning
            right_sums = [Fraction(0)]
            for y in right:
                ry = recip[y]
                # If ry alone exceeds target, no sum containing ry can work
                if ry > target:
                    continue
                new_sums = []
                for s in right_sums:
                    ns = s + ry
                    if ns <= target:
                        new_sums.append(ns)
                right_sums.extend(new_sums)

            right_set = set(right_sums)

            # Quick check: is target in right_set?
            if target in right_set:
                valid_cache[key] = False
                return False

            # Build left subset sums and check
            left_sums = [Fraction(0)]
            found = False
            for y in left:
                ry = recip[y]
                if ry > target:
                    continue
                new_sums = []
                for s in left_sums:
                    ns = s + ry
                    if ns <= target:
                        new_sums.append(ns)
                        if (target - ns) in right_set:
                            found = True
                            break
                if found:
                    break
                left_sums.extend(new_sums)
            
            if found:
                valid_cache[key] = False
                return False
                
        valid_cache[key] = True
        return True

    def local_search_improve(S):
        """
        Repeatedly add any element of [1..24] \ S that keeps the set valid,
        restarting after each successful addition (adding an element can open
        new possibilities for further additions).
        Also includes swap passes: 1-for-1, 2-for-1, 1-for-2 to escape local optima.
        """
        S = set(S)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Phase A: try to add elements
            for x in range(1, 25):
                if x not in S and is_valid(S | {x}):
                    S.add(x)
                    improved = True
                    break  # restart the pass
            # Phase B: if no addition worked, try 1-for-1 swaps
            if not improved:
                S_list = list(S)
                for x in S_list:
                    if time.time() > deadline:
                        break
                    for y in range(1, 25):
                        if y not in S:
                            new_S = (S - {x}) | {y}
                            if is_valid(new_S):
                                S = new_S
                                improved = True
                                break
                    if improved:
                        break
            # Phase C: try 2-for-1 swaps (remove 2, add 1)
            if not improved:
                S_list = list(S)
                for i in range(len(S_list)):
                    if time.time() > deadline:
                        break
                    for j in range(i + 1, len(S_list)):
                        if time.time() > deadline:
                            break
                        x1, x2 = S_list[i], S_list[j]
                        for y in range(1, 25):
                            if y not in S:
                                new_S = (S - {x1, x2}) | {y}
                                if is_valid(new_S):
                                    S = new_S
                                    improved = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
            # Phase D: try 1-for-2 swaps (remove 1, add 2)
            if not improved:
                S_list = list(S)
                for x in S_list:
                    if time.time() > deadline:
                        break
                    for y1 in range(1, 25):
                        if y1 not in S:
                            for y2 in range(y1 + 1, 25):
                                if y2 not in S:
                                    new_S = (S - {x}) | {y1, y2}
                                    if is_valid(new_S):
                                        S = new_S
                                        improved = True
                                        break
                            if improved:
                                break
                    if improved:
                        break
        return S

    def greedy_build(order):
        S = set()
        for x in order:
            if time.time() > deadline:
                break
            if x not in S and is_valid(S | {x}):
                S.add(x)
        return S

    def find_violation_details(S):
        """
        For set S, find all x in S where 1/x is a sum of reciprocals of a subset of S\{x}.
        Returns list of (x, subset) pairs where subset is a frozenset of elements from S\{x}
        whose reciprocals sum to 1/x.
        """
        S_list = sorted(S)
        violations = []
        for x in S_list:
            target = recip[x]
            others = [y for y in S_list if y != x]
            if not others:
                continue
            m = len(others)
            half = m // 2
            left = others[:half]
            right = others[half:]

            # Build right subset sums with tracking
            right_sums = [(Fraction(0), frozenset())]
            for y in right:
                ry = recip[y]
                if ry > target:
                    continue
                new_sums = []
                for (s, subset) in right_sums:
                    ns = s + ry
                    if ns <= target:
                        new_sums.append((ns, subset | {y}))
                right_sums.extend(new_sums)

            # Check if target is in right sums
            for (s, subset) in right_sums:
                if s == target:
                    violations.append((x, subset))
                    break
            else:
                # Build left subset sums and check
                left_sums = [(Fraction(0), frozenset())]
                for y in left:
                    ry = recip[y]
                    if ry > target:
                        continue
                    new_sums = []
                    for (s, subset) in left_sums:
                        ns = s + ry
                        if ns <= target:
                            new_sums.append((ns, subset | {y}))
                    left_sums.extend(new_sums)
                
                # Check combinations
                right_dict = {}
                for (s, subset) in right_sums:
                    if s not in right_dict:
                        right_dict[s] = subset
                
                for (ls, lsubset) in left_sums:
                    complement = target - ls
                    if complement in right_dict:
                        violations.append((x, lsubset | right_dict[complement]))
                        break
        
        return violations

    def repair_full():
        """Start from the full set; remove elements that cause a violation.
        Tries multiple strategies: remove the target x, remove an element from
        the violating subset, or remove the largest element in the violating subset."""
        best_repaired = set(range(1, 25))
        best_size = 0
        
        strategies = ['target', 'subset_smallest', 'subset_largest', 'subset_first']
        
        for strategy in strategies:
            if time.time() > deadline:
                break
            current = set(range(1, 25))
            for _ in range(100):
                if time.time() > deadline:
                    break
                if is_valid(current):
                    break
                violations = find_violation_details(current)
                if not violations:
                    break
                
                x, subset = violations[0]
                
                if strategy == 'target':
                    to_remove = x
                elif strategy == 'subset_smallest':
                    to_remove = min(subset) if subset else x
                elif strategy == 'subset_largest':
                    to_remove = max(subset) if subset else x
                elif strategy == 'subset_first':
                    to_remove = sorted(subset)[0] if subset else x
                else:
                    to_remove = x
                
                current.discard(to_remove)
            
            if len(current) > best_size:
                best_size = len(current)
                best_repaired = set(current)
        
        return best_repaired

    def update_best(S):
        nonlocal best_set, best_size
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)

    best_set = set()
    best_size = 0

    # Phase 1: repair from the full set
    random.seed(12345)
    S = repair_full()
    S = local_search_improve(S)
    update_best(S)
    
    # Phase 1b: try starting from subsets of size 18-24 and repairing
    for start_size in [20, 19, 18, 17, 16]:
        if time.time() > deadline:
            break
        for _ in range(5):
            if time.time() > deadline:
                break
            # Random subset of given size
            rand_set = set(random.sample(range(1, 25), start_size))
            # Repair: remove violators using the improved strategy
            current = set(rand_set)
            for _ in range(50):
                if time.time() > deadline:
                    break
                if is_valid(current):
                    break
                violations = find_violation_details(current)
                if not violations:
                    break
                x, subset = violations[0]
                # Remove smallest element from the violating subset
                to_remove = min(subset) if subset else x
                current.discard(to_remove)
            
            S = local_search_improve(current)
            update_best(S)

    # Phase 2: greedy builds from diverse orders, then local-search to fill in
    base = list(range(1, 25))
    orders = [
        base,
        list(reversed(base)),
        [1, 24, 2, 23, 3, 22, 4, 21, 5, 20, 6, 19, 7, 18, 8, 17, 9, 16, 10, 15, 11, 14, 12, 13],
        [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
        [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        # More structured orders
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
        [24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
    ]
    for _ in range(30):
        if time.time() > deadline:
            break
        o = list(base)
        random.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_build(order)
        S = local_search_improve(S)
        update_best(S)

    # Fallback: a singleton is always valid
    if not best_set:
        best_set = {24}

    return list(best_set)
