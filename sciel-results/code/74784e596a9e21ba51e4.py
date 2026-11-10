"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from functools import reduce
import random

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    start_time = time.time()
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    def check_dissociated(elements):
        """Check if all subset sums of reciprocals are distinct using integer arithmetic."""
        k = len(elements)
        if k == 0:
            return True
        
        # Compute LCM of all elements
        L = reduce(lcm, elements)
        
        # Convert reciprocals to integers: 1/x -> L/x
        int_vals = [L // x for x in elements]
        
        # Generate all subset sums and check for duplicates
        sums = {0}
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)
    
    def check_dissociated_fast(elements):
        """Faster check using incremental subset sum tracking with early termination."""
        k = len(elements)
        if k == 0:
            return True
        
        L = reduce(lcm, elements)
        int_vals = [L // x for x in elements]
        
        # Use set-based approach with early collision detection
        sums = {0}
        for v in int_vals:
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums:
                    return False
                new_sums.add(ns)
            sums.update(new_sums)
        
        return len(sums) == (1 << k)
    
    def check_dissociated_mitm(elements):
        """Meet-in-the-middle check for larger sets (k > 16)."""
        k = len(elements)
        if k <= 16:
            return check_dissociated_fast(elements)
        if k == 0:
            return True
        
        L = reduce(lcm, elements)
        int_vals = [L // x for x in elements]
        
        # Split into two halves
        mid = k // 2
        left = int_vals[:mid]
        right = int_vals[mid:]
        
        # Compute all subset sums for left half
        left_sums = {0}
        for v in left:
            new_sums = set()
            for s in left_sums:
                new_sums.add(s + v)
            left_sums.update(new_sums)
        
        if len(left_sums) != (1 << mid):
            return False
        
        # Compute all subset sums for right half
        right_sums = {0}
        for v in right:
            new_sums = set()
            for s in right_sums:
                new_sums.add(s + v)
            right_sums.update(new_sums)
        
        if len(right_sums) != (1 << (k - mid)):
            return False
        
        # Check for collisions: left_sum + right_sum should be unique
        # Total expected: 2^k = (1<<mid) * (1<<(k-mid))
        # We need to check that all sums left_s + right_s are distinct
        # This is equivalent to: for each left_s, the set {left_s + r for r in right_sums} 
        # should not intersect with {left_s' + r' for r' in right_sums} for any other left_s'
        # Simpler: just generate all and check size
        all_sums = set()
        for ls in left_sums:
            for rs in right_sums:
                s = ls + rs
                if s in all_sums:
                    return False
                all_sums.add(s)
        
        return len(all_sums) == (1 << k)
    
    def try_greedy_with_sums(order, current_sums, current_set):
        """Greedy build maintaining current_sums for efficiency."""
        current = list(current_set)
        sums = set(current_sums)
        
        for elem in order:
            if elem in current:
                continue
            L = reduce(lcm, current + [elem]) if current else elem
            if current:
                # Recompute int_vals with new LCM
                int_vals = [L // x for x in current + [elem]]
                new_sums = {0}
                for v in int_vals:
                    added = set()
                    for s in new_sums:
                        ns = s + v
                        if ns in new_sums:
                            return current, sums  # collision, stop
                        added.add(ns)
                    new_sums.update(added)
                if len(new_sums) != (1 << (len(current) + 1)):
                    return current, sums
                current.append(elem)
                sums = new_sums
            else:
                current.append(elem)
                sums = {0, L}
        
        return current, sums
    
    candidates = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # Generate diverse orderings
    orderings = [
        sorted(candidates, reverse=True),  # Largest first
        sorted(candidates),                 # Smallest first
        [22, 1, 21, 2, 20, 3, 19, 4, 18, 5, 17, 6, 16, 7, 15, 8, 14, 9, 13, 10, 12, 11],
        [2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 4, 8, 16, 3, 6, 12, 5, 10, 20, 7, 14, 21, 9, 18, 11, 22, 13, 15, 17, 19],
        [16, 12, 8, 4, 2, 20, 15, 10, 5, 22, 11, 6, 3, 18, 9, 7, 14, 21, 13, 19, 17, 1],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        
        current = []
        for elem in order:
            if time.time() > deadline:
                break
            if elem in current:
                continue
            test_set = current + [elem]
            if check_dissociated_fast(test_set):
                current = test_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Randomized greedy attempts
    random.seed(42)
    num_random_attempts = 200
    for _ in range(num_random_attempts):
        if time.time() > deadline - 10:
            break
        
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        current = []
        for elem in shuffled:
            if time.time() > deadline - 10:
                break
            if elem in current:
                continue
            test_set = current + [elem]
            if check_dissociated_fast(test_set):
                current = test_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Branch-and-bound local search: try to add elements
    if best_set and time.time() < deadline - 5:
        remaining = [x for x in candidates if x not in best_set]
        remaining.sort(reverse=True)
        
        for elem in remaining:
            if time.time() > deadline - 5:
                break
            test_set = best_set + [elem]
            if check_dissociated_fast(test_set):
                best_set = test_set
                best_size = len(best_set)
        
        # Swap improvement rounds
        for _ in range(5):
            if time.time() > deadline - 5:
                break
            progress = False
            for i, elem in enumerate(best_set):
                if time.time() > deadline - 5:
                    break
                for new_elem in candidates:
                    if new_elem in best_set:
                        continue
                    candidate_set = best_set[:i] + [new_elem] + best_set[i+1:]
                    candidate_set.sort()
                    if check_dissociated_fast(candidate_set):
                        best_set = candidate_set
                        progress = True
                        # Try to add more after swap
                        for add_elem in candidates:
                            if time.time() > deadline - 5:
                                break
                            if add_elem in best_set:
                                continue
                            test_set = best_set + [add_elem]
                            if check_dissociated_fast(test_set):
                                best_set = test_set
                        break
                if progress:
                    break
            if not progress:
                break
    
    # Final polish
    if time.time() < deadline - 3:
        for elem in sorted(candidates, reverse=True):
            if time.time() > deadline - 3:
                break
            if elem in best_set:
                continue
            test_set = best_set + [elem]
            if check_dissociated_fast(test_set):
                best_set = test_set
    
    if not best_set:
        best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
# EVOLVE-BLOCK-END
