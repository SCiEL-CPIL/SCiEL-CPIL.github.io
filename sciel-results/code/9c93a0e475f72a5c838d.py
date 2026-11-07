import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    # Helper to check if adding x to current sums is valid
    # Returns (valid, new_sums_set)
    def try_add(sums_set, x):
        # 1. Generate new sums
        new_sums = set()
        new_sums.add(x)
        for s in sums_set:
            new_sums.add(s + x)
        
        # 2. Check for duplicate sums (subset sum collision)
        if sums_set & new_sums:
            return False, None
            
        all_sums = sums_set | new_sums
        
        # 3. Check divisibility
        # We only need to check divisibility involving the new sums.
        # For each s in new_sums:
        #   a) Does any multiple of s exist in all_sums? (2s, 3s, ...)
        #   b) Does any divisor of s exist in all_sums? (s/2, s/3, ...)
        
        # Optimization: 
        # If s is large, 2s might be > max(all_sums), so no multiples.
        # If s is small, many divisors/multiples to check.
        
        max_sum = max(all_sums)
        
        # Create a fast lookup for all_sums
        # Since max_sum <= 256 * 30 = 7680, a boolean array is fast.
        # But we are doing this many times. 
        # Using a set is O(1) average.
        
        for s in new_sums:
            # Check multiples
            multiple = 2 * s
            while multiple <= max_sum:
                if multiple in all_sums:
                    return False, None
                multiple += s
                
            # Check divisors
            # s is divisible by d if s % d == 0 and d in all_sums
            # We can iterate d from 1 to sqrt(s)
            # Or just check if s is a multiple of any existing sum?
            # Actually, if d divides s, then s = k*d. 
            # We already checked if k*s is in all_sums for s in new_sums.
            # But here we are checking if s is a multiple of d (where d is in all_sums).
            # This is equivalent to checking if d is a divisor of s.
            
            # Efficient divisor check:
            # Iterate i from 2 to int(sqrt(s))
            # If s % i == 0, check if i in all_sums or s//i in all_sums.
            # Note: 1 is always a divisor, but 1 is not in sums unless x=1.
            # If 1 is in sums, then 1 divides everything. So if 1 is in sums, we can't add anything else?
            # Wait, if 1 is in sums, then for any other sum s, 1 divides s. So the set must be {1} or empty?
            # No, if S={1}, sums={1}. Valid.
            # If S={1,2}, sums={1,2,3}. 1 divides 2 and 3. Invalid.
            # So if 1 is in the set of sums, no other sum can exist.
            # This implies we should avoid 1 if we want size > 1.
            
            # Let's just check divisors properly.
            limit = int(s**0.5)
            for i in range(2, limit + 1):
                if s % i == 0:
                    if i in all_sums:
                        return False, None
                    if s // i in all_sums:
                        return False, None
                        
        return True, all_sums

    best_set = []
    best_size = 0
    
    candidates = list(range(1, 257))
    
    # Strategy 1: Descending
    for x in sorted(candidates, reverse=True):
        if time.time() > deadline:
            break
        if not best_set:
            sums_set = set()
        else:
            sums_set = set()
            # Recalculate sums for best_set? 
            # No, we should maintain the sums_set for the current candidate being built.
            pass
            
    # Let's structure the search properly.
    
    def greedy_search(order):
        current_set = []
        current_sums = set()
        for x in order:
            if time.time() > deadline:
                break
            valid, new_sums = try_add(current_sums, x)
            if valid:
                current_set.append(x)
                current_sums = new_sums
        return current_set
    
    # Run multiple greedy passes with different orders
    
    # 1. Descending
    s1 = greedy_search(sorted(candidates, reverse=True))
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = s1
        
    # 2. Ascending
    s2 = greedy_search(sorted(candidates))
    if len(s2) > best_size:
        best_size = len(s2)
        best_set = s2
        
    # 3. Primes first
    def is_prime(n):
        if n < 2: return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0: return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    composites = [x for x in candidates if not is_prime(x)]
    s3 = greedy_search(primes + composites)
    if len(s3) > best_size:
        best_size = len(s3)
        best_set = s3
        
    # 4. Randomized restarts
    # Bias towards larger numbers
    for i in range(200):
        if time.time() > deadline - 2.0:
            break
            
        # Create a weighted random order
        # Higher numbers have higher weight
        weights = [x for x in candidates]
        random_order = random.sample(candidates, len(candidates))
        
        # To bias towards larger, we can sort by random weight + value
        # Or just shuffle. Simple shuffle is often enough if we do many restarts.
        
        s = greedy_search(random_order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    # 5. Local Search / Swap
    # Try to improve best_set by swapping elements
    if best_set and time.time() < deadline - 5.0:
        current_best = list(best_set)
        current_sums = set()
        # Recalculate sums for current_best
        for x in current_best:
            _, current_sums = try_add(current_sums, x) # This is inefficient, just compute directly
        # Actually, let's just recompute sums for the final set
        def get_sums(S):
            sums = set()
            for x in S:
                new_sums = {x + s for s in sums}
                new_sums.add(x)
                sums |= new_sums
            return sums
            
        current_sums = get_sums(current_best)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            used = set(current_best)
            # Try removing each element and replacing with a better one
            for i in range(len(current_best)):
                if time.time() > deadline - 2.0:
                    break
                trial_set = current_best[:i] + current_best[i+1:]
                trial_sums = get_sums(trial_set)
                
                # Find a candidate x not in used that can be added
                # Prefer larger x?
                for x in sorted(candidates, reverse=True):
                    if x in used:
                        continue
                    valid, new_sums = try_add(trial_sums, x)
                    if valid:
                        new_set = trial_set + [x]
                        # Check if this new set is "better"
                        # For now, we just accept if it's the same size or larger
                        # But we want to maximize size. 
                        # If we swap, size is same. 
                        # We should only swap if it allows adding MORE elements later?
                        # This is complex. 
                        # Simple heuristic: If we can swap and then add another element, it's good.
                        # Let's just do a simple swap if it results in a valid set of same size, 
                        # hoping that the "structure" is better for future additions?
                        # Actually, if size is same, it doesn't help directly.
                        # Let's try to find a set of size best_size + 1.
                        
                        # Better approach: 
                        # If we found a valid set of size K, try to extend it.
                        # The greedy search already tries to extend.
                        # Local search is tricky here.
                        
                        # Let's just break and rely on greedy restarts.
                        break
                # End for x
            # End for i
        # End while improved
        
    # Final validation
    if not best_set:
        return [11, 13, 17]
        
    # Ensure types
    return [int(x) for x in best_set]

# Run a quick self-test in head
# try_add logic:
# sums_set = {2, 3, 5} (from {2,3})
# x = 7
# new_sums = {7, 9, 10, 12}
# all_sums = {2, 3, 5, 7, 9, 10, 12}
# Check 7: multiples 14, 21... none in all_sums. divisors: 1 (not in sums). OK.
# Check 9: multiples 18... none. divisors: 3 (in sums). 9 % 3 == 0. 3 in all_sums. Return False.
# Correct, because 3 divides 9.

# The logic holds.

# One optimization: 
# In `try_add`, checking divisors for every new sum is $O(\sqrt{s})$.
# Checking multiples is $O(\text{max\_sum}/s)$.
# This is fast enough for small sets.

# Let's add a "Sparse" heuristic: Numbers with few divisors are better.
# e.g., primes.

# The current code looks solid.

if __name__ == "__main__":
    res = construct()
    print(res)
