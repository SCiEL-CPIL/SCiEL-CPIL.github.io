import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 seconds margin before 90s kill
    
    # Precompute divisors for numbers up to max possible sum
    # Max sum for 10 elements of 256 is 2560. Let's go up to 3000 to be safe.
    MAX_SUM = 3000
    divisors = [[] for _ in range(MAX_SUM + 1)]
    for i in range(1, MAX_SUM + 1):
        for j in range(i, MAX_SUM + 1, i):
            divisors[j].append(i)
    
    # Helper to check if a new set of sums is valid given existing sums
    # existing_sums: set of sums
    # new_sums: set of sums involving the new element
    # Returns True if valid, False otherwise
    def is_valid_addition(existing_sums, new_sums):
        # Check for duplicates
        if existing_sums & new_sums:
            return False
        
        # Check divisibility:
        # 1. Any s_new in new_sums must not be divisible by any s_old in existing_sums
        # 2. Any s_old in existing_sums must not be divisible by any s_new in new_sums
        # 3. Any pair in new_sums must not divide each other
        
        # Efficient check for 1 & 2:
        for s_new in new_sums:
            # Check if s_new is divisible by any s_old
            # Iterate through divisors of s_new
            for d in divisors[s_new]:
                if d in existing_sums:
                    return False
            
            # Check if s_new divides any s_old
            # Iterate through multiples of s_new
            # Multiples are s_new, 2*s_new, ...
            # We can just check if any multiple is in existing_sums
            # But existing_sums is a set, so lookup is O(1).
            # How many multiples? MAX_SUM / s_new.
            # If s_new is small, this is many. But s_new is a sum of elements from 1..256.
            # Smallest s_new is min(S). If min(S) is large, few multiples.
            # If min(S) is small, many multiples, but then we likely fail early.
            for m in range(s_new * 2, MAX_SUM + 1, s_new):
                if m in existing_sums:
                    return False
        
        # Check divisibility within new_sums
        new_sums_list = list(new_sums)
        for i in range(len(new_sums_list)):
            a = new_sums_list[i]
            for j in range(i + 1, len(new_sums_list)):
                b = new_sums_list[j]
                if a > b:
                    if a % b == 0:
                        return False
                else:
                    if b % a == 0:
                        return False
                        
        return True

    def get_new_sums(current_set, x):
        """Generate new sums formed by adding x to subsets of current_set, plus x itself."""
        sums = set()
        sums.add(x)
        # We need the existing sums to form x + s
        # But we don't have the existing sums passed here in this signature.
        # Let's refactor: pass existing sums or compute them.
        # Actually, in the main loop, we have current_sums.
        pass

    def can_add(current_set, x, current_sums):
        # Generate new sums
        new_sums = set()
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(x + s)
        
        if not is_valid_addition(current_sums, new_sums):
            return False, None
            
        return True, new_sums

    candidates = list(range(1, 257))
    best_set = []
    best_size = 0
    
    # Initial greedy: Try descending order
    # Large numbers are less likely to divide others
    for x in sorted(candidates, reverse=True):
        valid, new_sums = can_add(best_set, x, set() if not best_set else get_all_sums(best_set))
        # Wait, we need to maintain current_sums in the loop.
        
    # Let's do a proper search loop
    def get_all_sums(S):
        sums = set()
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += S[i]
            sums.add(s)
        return sums

    max_restarts = 2000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Randomize order
        # Bias towards larger numbers?
        # Let's try a mix: 50% descending, 50% random
        if restart % 2 == 0:
            ordered = sorted(candidates, reverse=True)
        else:
            ordered = candidates.copy()
            random.shuffle(ordered)
            
        current_set = []
        current_sums = set()
        
        for x in ordered:
            valid, new_sums = can_add(current_set, x, current_sums)
            if valid:
                current_set.append(x)
                current_sums = current_sums.union(new_sums)
                
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set.copy()
        
        # Local search: Try to improve best_set
        if time.time() > deadline - 10:
            break
            
        if len(best_set) > 0:
            # Try swapping elements in best_set to allow adding more
            improved = False
            # We'll do a few iterations of local search
            for _ in range(5):
                if time.time() > deadline - 5:
                    break
                # Try removing one element and adding two? Or just adding one?
                # If we have size K, we want K+1.
                # Try adding a new element to best_set.
                # If it fails, try swapping one element in best_set with a different one, 
                # and then see if we can add a third.
                
                # Simpler: Just try to add any remaining element to best_set
                used = set(best_set)
                remaining = [x for x in candidates if x not in used]
                random.shuffle(remaining)
                
                # Try adding to a copy of best_set
                trial_set = best_set.copy()
                trial_sums = get_all_sums(trial_set)
                
                for x in remaining:
                    if time.time() > deadline - 5:
                        break
                    valid, new_sums = can_add(trial_set, x, trial_sums)
                    if valid:
                        new_set = trial_set + [x]
                        if len(new_set) > best_size:
                            best_size = len(new_set)
                            best_set = new_set
                            improved = True
                            break # Restart local search with new best
                
                if not improved:
                    break

    # Ensure return type is list of ints
    return [int(x) for x in best_set]
