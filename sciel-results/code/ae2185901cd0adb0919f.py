import math
import time
import random

def construct():
    n = 255
    start_time = time.time()
    deadline = start_time + 85  # 5-second safety margin
    
    # Prime factors of 255
    # 255 = 3 * 5 * 17
    
    # Baseline: all multiples of 3
    base_set = set(range(3, n + 1, 3))
    best_set = set(base_set)
    best_size = len(best_set)
    
    # All candidates: numbers in [1, n] that share a factor with 255
    # (must be divisible by 3, 5, or 17 to have gcd > 1 with 255)
    candidates = []
    for i in range(1, n + 1):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    # Try to build a larger set using greedy + local search
    # A set is valid if all pairs have gcd > 1
    
    def is_valid_set(s):
        """Check if all pairs in set s have gcd > 1"""
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if math.gcd(lst[i], lst[j]) == 1:
                    return False
        return True
    
    def can_add(s, x):
        """Check if x can be added to set s while maintaining validity"""
        for y in s:
            if math.gcd(x, y) == 1:
                return False
        return True
    
    # Greedy: try adding elements not in base_set
    current_set = set(base_set)
    
    # Sort candidates by some heuristic (e.g., number of prime factors shared with set)
    # Try adding each candidate
    for x in candidates:
        if x in current_set:
            continue
        if can_add(current_set, x):
            current_set.add(x)
    
    if len(current_set) > best_size:
        best_set = set(current_set)
        best_size = len(current_set)
    
    # Local search: try swapping elements
    # For each element in the set, try removing it and adding a different one
    for iteration in range(1000):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing one element and adding another
        for remove_candidate in list(best_set):
            if remove_candidate == n:  # Must keep 255
                continue
            
            temp_set = best_set - {remove_candidate}
            
            # Try adding each candidate not in temp_set
            for add_candidate in candidates:
                if add_candidate in temp_set:
                    continue
                if can_add(temp_set, add_candidate):
                    new_set = temp_set | {add_candidate}
                    if len(new_set) > best_size:
                        best_set = new_set
                        best_size = len(new_set)
                        improved = True
                        break
            if improved:
                break
        
        if not improved:
            # Try more aggressive: remove two, add one or two
            break
    
    # More sophisticated local search with random restarts
    random.seed(42)
    for restart in range(50):
        if time.time() > deadline:
            break
        
        # Start from a random valid subset
        # Use a random greedy construction
        current = set()
        if random.random() < 0.5:
            current.add(n)
        
        # Add elements greedily in random order
        shuffled = candidates[:]
        random.shuffle(shuffled)
        
        for x in shuffled:
            if x in current:
                continue
            if can_add(current, x):
                current.add(x)
        
        # Ensure 255 is in the set
        if n not in current:
            # Try to add 255
            if can_add(current, n):
                current.add(n)
            else:
                continue
        
        # Local improvement
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for remove_candidate in list(current):
                if remove_candidate == n:
                    continue
                temp_set = current - {remove_candidate}
                for add_candidate in candidates:
                    if add_candidate in temp_set:
                        continue
                    if can_add(temp_set, add_candidate):
                        new_set = temp_set | {add_candidate}
                        if len(new_set) > len(current):
                            current = new_set
                            improved = True
                            break
                if improved:
                    break
        
        if len(current) > best_size:
            best_set = set(current)
            best_size = len(current)
    
    # Final verification
    result = list(best_set)
    if n not in result:
        result.append(n)
    
    # Verify validity
    result_set = set(result)
    if not is_valid_set(result_set):
        # Fallback to baseline
        return sorted(base_set)
    
    return sorted(result_set)
