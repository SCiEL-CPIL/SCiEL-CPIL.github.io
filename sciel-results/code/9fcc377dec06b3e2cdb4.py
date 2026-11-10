"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
from itertools import combinations
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # LCM of 1..22 for integer representation of reciprocals
    # LCM(1,2,...,22) = 232792560
    LCM = 232792560
    
    def to_int(x):
        """Convert 1/x to integer representation using LCM."""
        return LCM // x
    
    def check_dissociated_int(ints):
        """Check if all subset sums are distinct using integer representation."""
        k = len(ints)
        if k == 0:
            return True
        sums = {0}
        for r in ints:
            new_sums = set()
            for s in sums:
                ns = s + r
                if ns in sums:
                    return False
                if ns in new_sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        return len(sums) == (1 << k)
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct."""
        ints = [to_int(x) for x in elements]
        return check_dissociated_int(ints)
    
    def try_order(order, current_best):
        """Try to build a dissociated set following the given order."""
        current = []
        current_ints = []
        for elem in order:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            test_ints = current_ints + [to_int(elem)]
            if check_dissociated_int(test_ints):
                current.append(elem)
                current_ints = test_ints
                if len(current) > len(current_best):
                    current_best = current[:]
        return current_best
    
    def try_order_with_backtrack(order, current_best, max_backtrack=3):
        """Try to build a dissociated set with limited backtracking."""
        current = []
        current_ints = []
        
        def backtrack(idx, depth):
            nonlocal current, current_ints
            if time.time() > deadline:
                return
            if idx >= len(order):
                if len(current) > len(current_best):
                    current_best = current[:]
                return
            elem = order[idx]
            if elem in current:
                backtrack(idx + 1, depth)
                return
            # Try adding elem
            test_ints = current_ints + [to_int(elem)]
            if check_dissociated_int(test_ints):
                current.append(elem)
                current_ints = test_ints
                backtrack(idx + 1, depth)
                current.pop()
                current_ints = test_ints[:-1]
            # Try skipping elem (only if depth < max_backtrack)
            if depth < max_backtrack:
                backtrack(idx + 1, depth + 1)
        
        backtrack(0, 0)
        return current_best
    
    candidates = list(range(1, 23))
    best_set = []
    
    # Strategy 1: Descending order (larger denominators first - smaller reciprocals)
    order1 = sorted(candidates, reverse=True)
    best_set = try_order(order1, best_set)
    
    # Strategy 2: Ascending order
    order2 = sorted(candidates)
    best_set = try_order(order2, best_set)
    
    # Strategy 3: Spaced out (alternating small and large)
    order3 = []
    for i in range(11):
        order3.append(i + 1)
        order3.append(22 - i)
    best_set = try_order(order3, best_set)
    
    # Strategy 4: Powers of 2 pattern (known to be dissociated)
    order4 = [1, 2, 4, 8, 16, 22, 21, 20, 19, 18, 17, 15, 14, 13, 12, 11, 10, 9, 7, 6, 5, 3]
    best_set = try_order(order4, best_set)
    
    # Strategy 5: Randomized attempts
    rng = random.Random(42)
    for attempt in range(10):
        if time.time() > deadline:
            break
        random_order = candidates[:]
        rng.shuffle(random_order)
        best_set = try_order(random_order, best_set)
    
    # Strategy 6: Start with known good set, try to add more
    if len(best_set) < 12:
        base = [1, 2, 4, 8, 16]
        remaining = [x for x in candidates if x not in base]
        # Try adding remaining elements in descending order
        current = base[:]
        current_ints = [to_int(x) for x in base]
        for elem in sorted(remaining, reverse=True):
            if time.time() > deadline:
                break
            test_ints = current_ints + [to_int(elem)]
            if check_dissociated_int(test_ints):
                current.append(elem)
                current_ints = test_ints
        if len(current) > len(best_set):
            best_set = current[:]
    
    # Strategy 7: Local improvement - try to swap elements to increase size
    if len(best_set) >= 5 and time.time() < deadline - 10:
        current = best_set[:]
        current_ints = [to_int(x) for x in current]
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 5:
                    break
                for j in range(len(candidates)):
                    if time.time() > deadline - 5:
                        break
                    elem = candidates[j]
                    if elem in current:
                        continue
                    # Try replacing current[i] with elem
                    new_ints = current_ints[:]
                    new_ints[i] = to_int(elem)
                    if check_dissociated_int(new_ints):
                        new_set = current[:]
                        new_set[i] = elem
                        # Check if this allows adding more
                        test_set = new_set[:]
                        test_ints = new_ints[:]
                        for add_elem in candidates:
                            if time.time() > deadline - 5:
                                break
                            if add_elem in test_set:
                                continue
                            t_ints = test_ints + [to_int(add_elem)]
                            if check_dissociated_int(t_ints):
                                test_set.append(add_elem)
                                test_ints = t_ints
                        if len(test_set) > len(current):
                            current = test_set[:]
                            current_ints = test_ints[:]
                            improved = True
                            break
                if improved:
                    break
    
    # Strategy 8: Enumerate subsets of size k starting from best+1
    target_size = len(best_set) + 1
    if target_size <= 18 and time.time() < deadline - 15:
        # C(22, k) for k up to 18: C(22,18)=C(22,4)=7315, very fast
        # C(22,17)=C(22,5)=26334, still fast
        # C(22,16)=C(22,6)=74613
        for k in range(target_size, min(19, 23)):
            if time.time() > deadline - 10:
                break
            found = False
            for combo in combinations(candidates, k):
                if time.time() > deadline - 10:
                    break
                ints = [to_int(x) for x in combo]
                if check_dissociated_int(ints):
                    best_set = list(combo)
                    found = True
                    break
            if found:
                break
    
    if not best_set:
        # Fallback to known good set
        best_set = [1, 2, 4, 8, 16, 22, 21, 20, 19, 18, 17]
    
    return best_set
# EVOLVE-BLOCK-END
