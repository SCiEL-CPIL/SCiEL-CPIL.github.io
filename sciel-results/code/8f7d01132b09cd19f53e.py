import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    TARGET = 123457
    
    # Precompute largest prime factor for all numbers up to TARGET
    # P[n] will store the largest prime factor of n
    # P[1] = 1
    max_n = TARGET
    lp = [1] * (max_n + 1)
    
    # Sieve for largest prime factor
    # Initialize lp[i] = i for all i, then for each prime p, update multiples
    # A more efficient way:
    # lp[n] = largest prime factor of n
    # We can iterate p from 2 to max_n. If lp[p] == p (i.e., p is prime), 
    # then for all multiples m of p, lp[m] = p.
    # Since we iterate p in increasing order, the last prime that divides m will be the largest.
    # So we can just do:
    for i in range(2, max_n + 1):
        if lp[i] == 1: # i is prime
            for j in range(i, max_n + 1, i):
                lp[j] = i
                
    # Now lp[n] holds the largest prime factor of n.
    # lp[1] is 1.
    
    best_value = float('inf')
    best_pair = (1, TARGET - 1)
    
    # Scan all possible splits
    # a from 1 to TARGET - 1
    # b = TARGET - a
    # cost = max(lp[a], lp[b])
    
    for a in range(1, TARGET):
        b = TARGET - a
        cost = max(lp[a], lp[b])
        if cost < best_value:
            best_value = cost
            best_pair = (a, b)
            
        # Optional: early termination if we hit 2? 
        # If best_value becomes 2, it's the absolute minimum possible for positive integers > 1.
        # But P(1)=1. If a=1, b=123456, P(b)=643.
        # If we can get both to be powers of 2, cost is 2.
        # 123457 is odd. a+b=odd => one is even, one is odd.
        # If a is power of 2, b is odd. P(b) must be 2 for cost 2? No, b is odd, so P(b) is odd prime >= 3.
        # So cost 2 is impossible.
        # Cost 3? a is 2-smooth, b is 3-smooth.
        # If best_value == 3, we can stop?
        # No, we just want the minimum.
        
        if time.time() > deadline:
            break
            
    return list(best_pair)
