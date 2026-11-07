import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    def is_valid_sums(sums):
        """Check if no two distinct sums divide each other."""
        if len(sums) <= 1:
            return True
        sum_list = sorted(sums)
        max_sum = sum_list[-1]
        sum_set = sums  # already a set
        for s in sum_list:
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in sum_set:
                    return False
                multiple += s
        return True
    
    def greedy_insert(candidates_order, current_elements, current_sums, target_size):
        """Try to insert elements greedily in the given order."""
        for c in candidates_order:
            if len(current_elements) >= target_size:
                break
            # Check if c is already in current_elements
            if c in current_elements:
                continue
            
            # Compute new sums
            new_sums = set(current_sums)
            for s in current_sums:
                ns = s + c
                if ns in new_sums:
                    return current_elements, current_sums  # duplicate sum, skip
                new_sums.add(ns)
            new_sums.add(c)
            
            # Check divisibility
            if is_valid_sums(new_sums):
                current_elements = current_elements + [c]
                current_sums = new_sums
        
        return current_elements, current_sums
    
    def local_search(current_elements, current_sums, candidates_pool, target_size, time_limit):
        """Try to extend the set by swapping elements or adding new ones."""
        current_elements = list(current_elements)
        current_sums = set(current_sums)
        used_set = set(current_elements)
        
        # Try to add elements not in the set
        available = [c for c in candidates_pool if c not in used_set]
        random.shuffle(available)
        
        for c in available:
            if time.time() > time_limit:
                break
            if len(current_elements) >= target_size:
                break
            
            new_sums = set(current_sums)
            valid = True
            for s in current_sums:
                ns = s + c
                if ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            
            if valid:
                new_sums.add(c)
                if is_valid_sums(new_sums):
                    current_elements.append(c)
                    current_sums = new_sums
                    used_set.add(c)
        
        # Try swaps: remove one element, try to add two
        if len(current_elements) < target_size:
            for i in range(len(current_elements)):
                if time.time() > time_limit:
                    break
                removed = current_elements[i]
                temp_elements = current_elements[:i] + current_elements[i+1:]
                
                # Recompute sums without removed element
                temp_sums = set()
                for mask in range(1, 1 << len(temp_elements)):
                    s = 0
                    for j in range(len(temp_elements)):
                        if mask & (1 << j):
                            s += temp_elements[j]
                    temp_sums.add(s)
                
                # Try to add elements
                for c in available:
                    if c in used_set or c == removed:
                        continue
                    new_sums = set(temp_sums)
                    valid = True
                    for s in temp_sums:
                        ns = s + c
                        if ns in new_sums:
                            valid = False
                            break
                        new_sums.add(ns)
                    if valid:
                        new_sums.add(c)
                        if is_valid_sums(new_sums):
                            # Try to add one more
                            for c2 in available:
                                if c2 in used_set or c2 == removed or c2 == c:
                                    continue
                                new_sums2 = set(new_sums)
                                valid2 = True
                                for s in new_sums:
                                    ns = s + c2
                                    if ns in new_sums2:
                                        valid2 = False
                                        break
                                    new_sums2.add(ns)
                                if valid2:
                                    new_sums2.add(c2)
                                    if is_valid_sums(new_sums2):
                                        current_elements = temp_elements + [c, c2]
                                        current_sums = new_sums2
                                        used_set = set(current_elements)
                                        break
                            break
        
        return current_elements, current_sums
    
    candidates = list(range(1, 257))
    best_set = []
    best_size = 0
    
    # Phase 1: Multiple greedy random restarts
    num_restarts = 500
    for attempt in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Different ordering strategies
        if attempt % 3 == 0:
            # Descending (large numbers first)
            order = sorted(candidates, reverse=True)
        elif attempt % 3 == 1:
            # Random
            order = candidates[:]
            random.shuffle(order)
        else:
            # Descending with some randomness
            order = sorted(candidates, reverse=True)
            # Swap some elements
            for _ in range(20):
                i = random.randint(0, len(order) - 1)
                j = random.randint(0, len(order) - 1)
                order[i], order[j] = order[j], order[i]
        
        current_elements, current_sums = greedy_insert(order, [], set(), 30)
        
        if len(current_elements) > best_size:
            best_size = len(current_elements)
            best_set = current_elements[:]
            if best_size >= 15:
                # Try local search to extend
                extended_elements, extended_sums = local_search(
                    current_elements, current_sums, candidates, 30, deadline - 5.0
                )
                if len(extended_elements) > best_size:
                    best_size = len(extended_elements)
                    best_set = extended_elements[:]
        
        if best_size >= 20:
            break
    
    # Phase 2: If we have a good set, try to extend it further
    if best_size > 0 and best_size < 30 and time.time() < deadline - 5.0:
        extended_elements, extended_sums = local_search(
            best_set, set(), candidates, 30, deadline - 3.0
        )
        if len(extended_elements) > best_size:
            best_size = len(extended_elements)
            best_set = extended_elements[:]
    
    # Final verification
    if best_set:
        # Verify the set is valid
        n = len(best_set)
        all_sums = set()
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += best_set[i]
            all_sums.add(s)
        
        if not is_valid_sums(all_sums):
            # Fallback to a known valid small set
            best_set = [31, 32, 34]
            best_size = 3
    
    # Ensure we return a valid non-empty set
    if not best_set:
        # Try a few small known-valid sets
        for test_set in [[31, 32, 34], [100, 101, 102], [100, 101], [100]]:
            n = len(test_set)
            all_sums = set()
            for mask in range(1, 1 << n):
                s = 0
                for i in range(n):
                    if mask & (1 << i):
                        s += test_set[i]
                all_sums.add(s)
            if is_valid_sums(all_sums):
                best_set = test_set
                break
    
    return best_set

# EVOLVE-BLOCK-END
