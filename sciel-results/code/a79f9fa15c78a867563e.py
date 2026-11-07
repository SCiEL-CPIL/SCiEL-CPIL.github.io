"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    N = 256
    candidates = list(range(2, N + 1))
    
    # Generate all primes
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in candidates if is_prime(x)]
    
    # Initial valid set: All primes
    S = set(primes)
    
    def is_valid(S):
        # Check if set S is valid
        # For all distinct a, b, c in S: a does not divide b*c
        lst = list(S)
        n = len(lst)
        for i in range(n):
            a = lst[i]
            for j in range(n):
                if i == j:
                    continue
                b = lst[j]
                for k in range(n):
                    if k == i or k == j:
                        continue
                    c = lst[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Heuristic check for adding/removing/swapping to speed up search
    # We will use local search.
    
    best_S = set(S)
    best_size = len(S)
    
    start_time = time.time()
    
    # Local Search: Try to swap elements or add elements
    # Since adding to a set of primes is hard, we focus on swaps.
    # Or start from a different base?
    
    # Let's try a randomized local search starting from the set of primes.
    # Move: Remove x, Add y (if valid)
    # Move: Remove x, Add y, Add z (if valid) -> increases size by 1
    
    # To make it faster, we precompute prime factors or just use the check.
    # The check is O(N^3). N=54. 54^3 = 157k. Fast enough for a few thousand iterations.
    
    # Optimization:
    # Instead of checking full validity every time, we can check incremental validity.
    # But for simplicity and robustness, we might just check the whole set if we do few iterations.
    # 60s limit. 157k ops is ~0.01s. We can do ~1000 checks per second?
    # Actually Python is slower. 157k inner loop iterations might take 0.1s.
    # So ~10 checks per second. That's not enough for random search.
    
    # We need a faster validity check or a smarter search.
    
    # Smarter approach:
    # The condition a | b*c is the bottleneck.
    # If we maintain the set, we can check if a new element 'a' can be added to S.
    # And if an element 'b' can be removed from S.
    
    def can_add(S, a):
        # Check if a can be added to S
        # 1. a | b*c for b,c in S, b!=c
        # 2. b | a*c for b,c in S, b!=c
        # 3. c | a*b for b,c in S, b!=c
        # Note: if a is in S, we can't add it.
        if a in S:
            return False
            
        lst = list(S)
        n = len(lst)
        # Check 1: a | b*c
        for i in range(n):
            b = lst[i]
            for j in range(n):
                if i == j:
                    continue
                c = lst[j]
                if (b * c) % a == 0:
                    return False
                
        # Check 2 & 3: 
        # For all b in S, b | a*c for some c in S, c!=b?
        # Actually, the condition is "For every THREE DISTINCT members a,b,c... a does not divide b*c".
        # This applies to the NEW set S' = S U {a}.
        # The triples involving a are:
        # (a, b, c) -> a | b*c (Checked above)
        # (b, a, c) -> b | a*c
        # (b, c, a) -> c | a*b (Same as b | a*c with b,c swapped)
        
        # So we need to check:
        # For all distinct b, c in S:
        # b | a*c is False
        # c | a*b is False
        
        for i in range(n):
            b = lst[i]
            for j in range(n):
                if i == j:
                    continue
                c = lst[j]
                if (a * c) % b == 0:
                    return False
                if (a * b) % c == 0:
                    return False
        return True

    def can_remove(S, a):
        # Removing an element always preserves validity of the remaining set?
        # Yes, because any valid triple in S\{a} was also a valid triple in S.
        # So removal is always safe.
        return True

    # Search Strategy:
    # 1. Start with S = Primes.
    # 2. Try to find a pair (x, y) such that removing x and adding y increases size? No, size stays same.
    # 3. Try to find x in S and y, z not in S such that (S \ {x}) U {y, z} is valid.
    #    This increases size by 1.
    
    # To do this efficiently:
    # Iterate through candidates for removal (x in S).
    # For each x, consider S' = S \ {x}.
    # Try to add two elements y, z from candidates \ S.
    # This is expensive.
    
    # Alternative:
    # Just try to add ONE element to S. If impossible, try to swap.
    # But we know we can't add to Primes.
    
    # Let's try a different starting point:
    # Maybe a set of composite numbers allows more density?
    # No, primes are the most "efficient" in terms of prime usage.
    
    # Let's stick to the local search on the set of primes, but optimize the check.
    
    # We will use a time limit.
    iterations = 0
    while time.time() - start_time < 50:
        iterations += 1
        # Randomly pick an element to remove
        if len(S) < 2:
            break
        x = random.choice(list(S))
        S_trial = S - {x}
        
        # Try to add 1 or 2 random candidates
        candidates_not_in_S = [c for c in candidates if c not in S_trial]
        
        # Try adding 2
        if len(candidates_not_in_S) >= 2:
            y, z = random.sample(candidates_not_in_S, 2)
            # Check if y can be added to S_trial
            if can_add(S_trial, y):
                S_temp = S_trial | {y}
                if can_add(S_temp, z):
                    S_new = S_temp | {z}
                    if len(S_new) > best_size:
                        best_size = len(S_new)
                        best_S = S_new
                        S = S_new
                        # Restart search with new best
                        continue
        
        # Try adding 1 (might allow future adds)
        if candidates_not_in_S:
            y = random.choice(candidates_not_in_S)
            if can_add(S_trial, y):
                S_new = S_trial | {y}
                if len(S_new) > best_size:
                    best_size = len(S_new)
                    best_S = S_new
                    S = S_new
                    continue
                # Even if size doesn't increase, keep S_new if it's "better" structurally?
                # No, just keep S for now to allow further exploration.
                # But if we don't update S, we might loop.
                # Let's update S to S_new to explore from there, even if size is same.
                S = S_new

    return list(best_S)
# EVOLVE-BLOCK-END
