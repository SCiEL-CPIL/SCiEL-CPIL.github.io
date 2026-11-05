import time
import random
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    n = 256
    # We want to maximize the size of a subset S of [1, n]
    # such that for all a < b <= c in S, a does not divide b+c.
    
    # Helper function to check if adding x to S is valid
    # S is a list or set. To speed up, we can precompute conflicts or just check on the fly.
    # Given n=256, the size of S is at most 256.
    # Checking a new element x against existing S:
    # For each pair (b, c) in S with b <= c and x < b:
    #   if (b+c) % x == 0: conflict.
    # This is O(|S|^2) per insertion check. |S| ~ 100-200. |S|^2 ~ 20,000.
    # 256 insertions * 20,000 checks = 5 million operations. Very fast.
    
    def is_valid_addition(x, S):
        # S is a list of elements.
        # We need to check all pairs (b, c) in S such that x < b <= c.
        # Note: Since we are adding x, x is not in S yet.
        # If x is larger than all elements in S, there are no b, c with x < b, so it's always valid.
        if not S:
            return True
        
        max_s = max(S)
        if x >= max_s:
            # x is the largest, so no b in S satisfies x < b.
            return True
        
        # We need to check pairs (b, c) from S where b > x.
        # Let S_gt = [y for y in S if y > x]
        # For all b, c in S_gt with b <= c, check if x divides b+c.
        # Actually, the condition is for ANY a < b <= c in the FINAL set.
        # If we add x, x becomes a new 'a' candidate, or x becomes a 'b' or 'c' candidate?
        # The condition is: For every a < b <= c in the set, a NOT divide b+c.
        
        # Case 1: x is the 'a'.
        # We need to ensure that for all b, c in S such that x < b <= c, x does not divide b+c.
        # Case 2: x is the 'b' or 'c'.
        # If x is b or c, it means there exists a < x in S such that a divides x+c (if x=b) or a divides b+x (if x=c).
        # Since the set is static until we add x, we must check if adding x creates any NEW violations.
        # Violations involving x:
        # 1. x is a: a=x. Need b,c in S, x < b <= c, x | (b+c).
        # 2. x is b: b=x. Need a,c in S, a < x <= c, a | (x+c).
        # 3. x is c: c=x. Need a,b in S, a < b <= x, a | (b+x).
        
        # Let's just check all triples involving x.
        # To be efficient, we can just iterate through S.
        
        # 1. x is a
        # Find all b, c in S with x < b <= c.
        # We can sort S for easier handling, but S is small.
        # Let's just loop.
        for i in range(len(S)):
            b = S[i]
            if b <= x:
                continue
            # b > x
            # Now check c in S such that b <= c
            for j in range(len(S)):
                c = S[j]
                if c < b:
                    continue
                # b <= c
                # Check if x divides b+c
                if (b + c) % x == 0:
                    return False
        
        # 2. x is b
        # Need a, c in S such that a < x <= c and a | (x+c)
        for i in range(len(S)):
            a = S[i]
            if a >= x:
                continue
            # a < x
            # Now check c in S such that x <= c
            for j in range(len(S)):
                c = S[j]
                if c < x:
                    continue
                # x <= c
                # Check if a divides x+c
                if (x + c) % a == 0:
                    return False
        
        # 3. x is c
        # Need a, b in S such that a < b <= x and a | (b+x)
        for i in range(len(S)):
            a = S[i]
            # a < b <= x implies a < x
            if a >= x:
                continue
            for j in range(len(S)):
                b = S[j]
                if b <= a or b > x:
                    continue
                # a < b <= x
                # Check if a divides b+x
                if (b + x) % a == 0:
                    return False
                    
        return True

    # Strategy: Greedy construction followed by local search improvement.
    # 1. Start with a good seed. The upper tail is a safe start.
    #    Let's try to construct a set by adding elements from 1 to n, or n to 1?
    #    Adding from large to small is often better because large numbers are less likely to divide sums of other large numbers?
    #    Actually, if x is large, it's hard for x to divide b+c unless b+c is a multiple of x.
    #    If x is small, it's easy for x to divide b+c.
    #    So adding large numbers first is "safe".
    
    S = []
    # Initialize with the upper tail as a strong baseline
    # Let's try to add numbers from n down to 1 if possible.
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_addition(x, S):
            S.append(x)
    
    current_size = len(S)
    
    # Local Search: Try to add more numbers or swap to increase size.
    # Since we added in descending order, we might have missed some small numbers that fit.
    # Also, we can try to remove a number to allow two others?
    # Simpler: Try to add any remaining number from 1 to n that wasn't added.
    
    # Let's do a few rounds of random restarts or perturbations.
    # Given the time limit, we can do multiple greedy runs with different orders or random shuffles.
    
    best_set = S
    best_size = current_size
    
    # Try a few different greedy orders
    # 1. Ascending (might fail early but let's see)
    # 2. Random
    # 3. Primality based?
    
    for attempt in range(20):
        if time.time() > deadline:
            break
        
        # Randomize the order of candidates
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        S_try = []
        for x in candidates:
            if time.time() > deadline:
                break
            if is_valid_addition(x, S_try):
                S_try.append(x)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]
        
        # Also try a "swap" improvement on best_set?
        # If we can't add, can we replace one element with two?
        # This is complex. Let's stick to greedy with randomization first.
        
        # Another strategy: Start with the best set found, and try to add one random element
        # that was rejected in the greedy pass, or remove one and add two?
        # For now, random greedy restarts are a solid "different strategy" from static tail.
        
        # Let's also try a "complement" approach?
        # No, let's try adding small numbers specifically to the best_set.
        # If we have a valid set, can we add x?
        # We already did this in the greedy pass for that specific order.
        
        # Let's try a specific heuristic: 
        # Keep the large numbers, try to insert small numbers.
        # This is effectively what the random shuffle might do, but let's be targeted.
        
        # Targeted: Take the current best_set (which likely contains large numbers).
        # Try to add any number from 1 to n not in best_set.
        S_targeted = best_set[:]
        for x in range(1, n + 1):
            if time.time() > deadline:
                break
            if x in S_targeted:
                continue
            if is_valid_addition(x, S_targeted):
                S_targeted.append(x)
        
        if len(S_targeted) > best_size:
            best_size = len(S_targeted)
            best_set = S_targeted[:]
            
    return best_set
# EVOLVE-BLOCK-END
