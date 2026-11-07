import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin
    
    # Initial best set
    best_set = [11, 13, 17]
    best_size = len(best_set)
    
    # Helper to check if x can be added to current_sums
    def can_add(current_sums, x):
        # Generate new sums
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(x + s)
        
        # 1. Check for duplicate sums (intersection)
        if current_sums & new_sums:
            return False, None

        max_sum_val = max(current_sums) if current_sums else 0
        max_new_val = max(new_sums)
        global_max = max(max_sum_val, max_new_val)
        
        # 2. Check divisibility
        # For each new sum, check against old sums and other new sums
        
        # Optimization: Precompute divisors? No, just check on the fly.
        
        for n in new_sums:
            # Check if any old sum divides n
            # Iterate divisors of n
            limit = int(n**0.5) + 1
            for d in range(2, limit):
                if n % d == 0:
                    if d in current_sums:
                        return False, None
                    other = n // d
                    if other in current_sums:
                        return False, None
            
            # Check if n divides any old sum
            # Iterate multiples of n
            m = 2 * n
            while m <= max_sum_val:
                if m in current_sums:
                    return False, None
                m += n
            
            # Check interactions within new_sums
            # We need to ensure no two distinct elements in new_sums divide each other.
            # Since we are iterating n in new_sums, we can check if any other n' in new_sums
            # is a multiple of n or a divisor of n.
            # To avoid double checking and self-checking, we can just check if any
            # multiple of n exists in new_sums (excluding n itself).
            # And if any divisor of n exists in new_sums (excluding n itself).
            
            # Check if n divides any other element in new_sums
            m = 2 * n
            while m <= max_new_val:
                if m in new_sums:
                    return False, None
                m += n
                
            # Check if any other element in new_sums divides n
            # We already checked divisors of n against current_sums.
            # Now check divisors of n against new_sums.
            limit = int(n**0.5) + 1
            for d in range(2, limit):
                if n % d == 0:
                    if d in new_sums:
                        return False, None
                    other = n // d
                    if other in new_sums:
                        return False, None
                            
        return True, new_sums

    # Randomized greedy search
    candidates = list(range(1, 257))
    
    # Try a few random restarts
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        random.shuffle(candidates)
        current_set = []
        current_sums = set()
        
        for x in candidates:
            if time.time() > deadline:
                break
                
            valid, new_sums = can_add(current_sums, x)
            if valid:
                current_set.append(x)
                current_sums = current_sums.union(new_sums)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
                    
                    # If we reach a very high size, we might break early?
                    # Theoretical max is around 15-20?
                    # If we hit 20, we can stop searching for larger.
                    if best_size >= 25:
                        # Unlikely, but safe
                        break

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
