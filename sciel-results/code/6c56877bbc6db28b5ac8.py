import time
import random
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute reciprocals
    reciprocals = {i: Fraction(1, i) for i in range(1, 23)}
    all_elements = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    def try_add(current_sums, element):
        """Check if element can be added; return new sums set or None."""
        r = reciprocals[element]
        new_sums = set(current_sums)
        for s in current_sums:
            ns = s + r
            if ns in new_sums:
                return None
            new_sums.add(ns)
        return new_sums
    
    def build_sums_set(elements):
        """Build the set of all subset sums for elements, return None if conflict."""
        sums = {Fraction(0, 1)}
        for e in elements:
            r = reciprocals[e]
            new_sums = set(sums)
            for s in sums:
                ns = s + r
                if ns in new_sums:
                    return None
                new_sums.add(ns)
            sums = new_sums
        return sums
    
    # Phase 1: Deterministic greedy with multiple orderings
    orderings = [
        list(range(1, 23)),           # ascending
        list(range(22, 0, -1)),       # descending
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 2, 3, 5, 7, 11, 13, 17, 19, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22],
        [1, 3, 2, 5, 4, 7, 6, 11, 10, 13, 12, 17, 16, 19, 18, 22, 21, 20, 15, 14, 9, 8],
        [2, 4, 1, 6, 3, 8, 5, 12, 7, 14, 9, 18, 10, 20, 11, 22, 13, 21, 15, 19, 16, 17],
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        current_set = []
        current_sums = {Fraction(0, 1)}
        for e in order:
            new_sums = try_add(current_sums, e)
            if new_sums is not None:
                current_set.append(e)
                current_sums = new_sums
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
    
    # Phase 2: Randomized restarts with local search
    while time.time() < deadline - 2:
        candidates = all_elements[:]
        random.shuffle(candidates)
        
        current_set = []
        current_sums = {Fraction(0, 1)}
        
        for e in candidates:
            if time.time() > deadline - 1:
                break
            new_sums = try_add(current_sums, e)
            if new_sums is not None:
                current_set.append(e)
                current_sums = new_sums
        
        # Local search: try to improve by removing one and adding others
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Try local improvement on the current set
        if time.time() < deadline - 3:
            improved = True
            while improved and time.time() < deadline - 2:
                improved = False
                remaining = [e for e in all_elements if e not in current_set]
                
                for i in range(len(current_set)):
                    if time.time() > deadline - 2:
                        break
                    
                    removed_elem = current_set[i]
                    test_set = current_set[:i] + current_set[i+1:]
                    test_sums = build_sums_set(test_set)
                    
                    if test_sums is None:
                        continue
                    
                    # Greedily add remaining elements
                    added_any = False
                    for e in remaining:
                        new_sums = try_add(test_sums, e)
                        if new_sums is not None:
                            test_sums = new_sums
                            test_set.append(e)
                            added_any = True
                    
                    if len(test_set) > len(current_set):
                        current_set = test_set
                        current_sums = test_sums
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = current_set[:]
                        break  # Restart inner loop
    
    # Phase 3: Aggressive final polishing on best_set
    if best_set and time.time() < deadline - 1:
        best_sums = build_sums_set(best_set)
        if best_sums is not None:
            # Try adding multiple elements iteratively
            changed = True
            while changed and time.time() < deadline - 1:
                changed = False
                for e in all_elements:
                    if e not in best_set:
                        new_sums = try_add(best_sums, e)
                        if new_sums is not None:
                            best_set.append(e)
                            best_sums = new_sums
                            best_size = len(best_set)
                            changed = True
                            break
            
            # Try swap-based improvement: remove one, add multiple
            if time.time() < deadline - 1:
                for i in range(len(best_set)):
                    if time.time() > deadline - 1:
                        break
                    test_set = best_set[:i] + best_set[i+1:]
                    test_sums = build_sums_set(test_set)
                    if test_sums is None:
                        continue
                    for e in all_elements:
                        if e not in test_set:
                            new_sums = try_add(test_sums, e)
                            if new_sums is not None:
                                test_sums = new_sums
                                test_set.append(e)
                    if len(test_set) > len(best_set):
                        best_set = test_set
                        best_sums = test_sums
                        best_size = len(best_set)
                        break
    
    if not best_set:
        return [1]
    
    # Final verification
    final_sums = build_sums_set(best_set)
    if final_sums is None:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    return best_set
