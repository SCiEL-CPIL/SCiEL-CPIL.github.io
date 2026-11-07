import random
import time
import sys

def construct():
    MAX_VAL = 256
    MAX_K = 30
    MAX_SUM = MAX_VAL * MAX_K
    
    # Precompute multiple masks
    # multiple_masks[s] has bits set at 2*s, 3*s, ... up to MAX_SUM
    multiple_masks = [0] * (MAX_SUM + 1)
    for s in range(1, MAX_SUM // 2 + 1):
        mask = 0
        m = 2 * s
        while m <= MAX_SUM:
            mask |= (1 << m)
            m += s
        multiple_masks[s] = mask

    def is_valid(bits, k):
        # Check for collisions (distinct sums)
        if bits.bit_count() != (1 << k) - 1:
            return False
        
        # Check for divisibility
        # Iterate through set bits in bits
        b = bits
        while b:
            lsb = b & -b
            s = lsb.bit_length() - 1
            if s > 0:
                if bits & multiple_masks[s]:
                    return False
            b ^= lsb
        return True

    best_set = []
    best_size = 0
    
    start_time = time.time()
    timeout = 55 # Leave buffer
    
    # Backtracking search
    # We'll try to find a set of size K.
    # Start with K=10, 12, etc? Or just DFS and track max?
    # DFS can be deep. Let's do a randomized DFS.
    
    def backtrack(current_set, bits, k, candidates, depth):
        nonlocal best_set, best_size
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
        
        # Pruning: if we've used up candidates or time
        if time.time() - start_time > timeout:
            return
        
        # If we reached a good size, we can stop this branch?
        # No, we want max. But if we are at K=15, finding 16 is hard.
        
        # Try adding each candidate
        # Heuristic: sort candidates? 
        # Randomizing the order of candidates helps avoid getting stuck.
        
        # To speed up, we can filter candidates that are "safe" immediately.
        # But checking safety requires the full is_valid check.
        
        # Let's just iterate through remaining candidates.
        # To avoid symmetric states, we can enforce an order?
        # No, the set is unordered, but the sums depend on the specific values.
        # We can enforce that we only pick candidates in increasing order to avoid permutations?
        # If we enforce increasing order, we miss sets like {10, 5} if we pick 10 then 5.
        # But {5, 10} is the same set.
        # So we can enforce that we only pick from candidates > last_picked?
        # This reduces the search space significantly (combinations vs permutations).
        
        last_picked = current_set[-1] if current_set else 0
        # Filter candidates to be > last_picked
        # This ensures we generate each subset only once in sorted order.
        
        for i in range(len(candidates)):
            x = candidates[i]
            if x <= last_picked:
                continue
                
            new_bits = bits | (bits << x) | (1 << x)
            
            # Quick check: did we create a collision?
            # The new sums are the old sums + x, and x itself.
            # A collision occurs if:
            # 1. x is already in bits (impossible if x > last_picked and bits only contains sums of previous smaller numbers? 
            #    Wait, sums can be larger than x. E.g. {1, 2} -> sums 1, 2, 3. Add 4. 4 is not in {1,2,3}. 
            #    But if we add 1 to {1, 2, 3, 4}? No, we are building from scratch.
            #    Since we enforce increasing order of elements, x is larger than all previous elements.
            #    However, subset sums of previous elements can be larger than x.
            #    Example: prev set {1, 2, 4}, sums {1,2,3,4,5,6,7}. Add x=8.
            #    New sums: 8, 9, 10, 12, 13, 14, 15.
            #    Collision if any new sum is in old bits.
            #    `new_bits` handles this via OR.
            #    To check collision fast: 
            #    `new_sums = (bits << x) | (1 << x)`
            #    If `new_sums & bits != 0`, there is a collision.
            #    Also, if `new_sums` has internal collisions?
            #    If `bits` had no collisions, `bits << x` has no collisions.
            #    `1 << x` is a single bit.
            #    Collision if `((bits << x) | (1 << x)) & bits != 0` OR `((bits << x) | (1 << x))` has internal duplicates?
            #    Internal duplicates in `bits << x` are impossible if `bits` had no duplicates.
            #    Duplicate between `bits << x` and `1 << x`? Impossible since x > 0.
            #    So collision check is just `((bits << x) | (1 << x)) & bits != 0`.
            
            new_part = (bits << x) | (1 << x)
            if new_part & bits:
                continue
                
            # Check divisibility
            # We only need to check if the NEW sums divide OLD sums or vice versa, or NEW divide NEW.
            # But `is_valid(new_bits, k+1)` is robust.
            # Is it fast enough?
            # `is_valid` iterates set bits. 
            # Let's try it.
            
            if is_valid(new_bits, k + 1):
                backtrack(current_set + [x], new_bits, k + 1, candidates, depth + 1)
                
            # Pruning: if we found a large set, maybe stop?
            # We can't stop easily in DFS without passing a flag.

    # Initial candidates
    # To find larger sets, we should probably start with larger numbers?
    # Or smaller?
    # Small numbers create many small sums which are more likely to divide other sums.
    # Large numbers create large sums which are less likely to divide anything (unless the other sum is a multiple, which is less likely for large numbers).
    # So, picking large numbers first might be better for validity.
    # However, we enforce increasing order in `backtrack`.
    # So the "first" picked number is the smallest in the set.
    # If we want to pick large numbers, we should start the candidate list with large numbers?
    # No, if we enforce `x > last_picked`, the first pick is the min of the set.
    # So we should try small mins? Or large mins?
    # If min is small (e.g., 1), then 1 divides everything. So 1 cannot be in the set.
    # So min must be > 1.
    # If min is 2, then 2 divides 4, 6, etc.
    
    # Let's just use all candidates 1..256.
    # Randomize the order of candidates to explore different branches?
    # Since we enforce increasing order, the order of `candidates` list determines the branching order.
    # If we shuffle `candidates`, we explore different "first elements" (min elements) in different orders.
    
    candidates = list(range(1, MAX_VAL + 1))
    
    # We can try multiple random shuffles of candidates
    while time.time() - start_time < timeout:
        random.shuffle(candidates)
        # Reset best for this attempt? No, keep global best.
        backtrack([], 0, 0, candidates, 0)
        
        # If we found a very large set, we might break?
        if best_size >= 20:
             break

    return [int(x) for x in best_set]
