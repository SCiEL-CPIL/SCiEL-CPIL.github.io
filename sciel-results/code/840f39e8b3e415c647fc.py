import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    TARGET = 123457
    
    # Step 1: Sieve of Eratosthenes to get primes up to TARGET
    # We need primes up to TARGET to fully factor any number up to TARGET
    # But actually, for the "smooth number" approach, we just need to iterate primes
    # and check if a pair (a, b) exists where both are p-smooth.
    # The largest prime factor of the optimal solution is the first prime p
    # such that there exist a, b <= TARGET, a+b=TARGET, and both a and b are p-smooth.
    
    # Let's generate primes up to a reasonable limit.
    # The trivial solution is (1, 123456). 123456 = 2^5 * 3 * 7 * 19 * 23.
    # So P(123456) = 23. Thus the answer is at most 23.
    # We only need to check primes up to 23 to find the minimum.
    # Let's generate primes up to 100 just to be safe, though 23 is likely the bound.
    
    limit = 100
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    # We will iterate through primes p in increasing order.
    # For each p, we check if there is a decomposition a+b=TARGET
    # such that both a and b have no prime factors > p.
    # This is equivalent to saying a and b are p-smooth.
    
    # Efficient way to check:
    # 1. Generate all p-smooth numbers <= TARGET.
    # 2. Check if any pair sums to TARGET.
    
    # Since p is small (likely <= 23), the set of p-smooth numbers is small.
    # We can generate them incrementally or just check for each p.
    
    # Let's try primes in order.
    # For each prime p, we need to know if there exist a, b such that:
    # a + b = TARGET
    # P(a) <= p and P(b) <= p
    
    # We can iterate through all possible a from 1 to TARGET-1.
    # For a given a, b = TARGET - a.
    # We need to check if P(a) <= p and P(b) <= p.
    
    # To make this fast, we can precompute P(n) for all n?
    # That was the original approach. It was O(TARGET * log(log(TARGET))) roughly.
    # The original code did this and found the answer.
    # The score was 58.45.
    # Let's see if we can do better or if the original was already optimal.
    
    # The original code computed lpf[n] for all n up to TARGET.
    # Then it scanned for min max(lpf[a], lpf[b]).
    # This is correct.
    # Why was the score 58.45?
    # 123457 is the target.
    # If the optimal P is 23, the score should be related to 23.
    # The score formula is seed/value.
    # If the seed program returned a bad answer, the score would be low.
    # If the seed program returned the optimal answer, the score would be high.
    # Wait, the prompt says "score 58.4545".
    # And "1.0 equals the seed program and above 1.0 is an improvement".
    # So 58.45 is a HUGE improvement over the seed.
    # This implies the seed was very bad.
    # And 58.45 is the current best.
    # Is it possible to do better than 58.45?
    # The value is max(P(a), P(b)).
    # If we find a pair with max(P(a), P(b)) = 23, and the seed had value V_seed,
    # Score = V_seed / 23.
    # If Score = 58.45, then V_seed = 58.45 * 23 ≈ 1344.
    # This seems plausible.
    
    # Can we get a lower value than 23?
    # We need to check if there is a pair with max(P(a), P(b)) < 23.
    # i.e., both a and b are 19-smooth (primes <= 19).
    # Or 17-smooth, etc.
    
    # Let's check p=19.
    # Are there 19-smooth numbers a, b such that a+b=123457?
    # Let's write a quick check in our head or use the code to verify.
    # The code below will do this check efficiently.
    
    # Strategy:
    # 1. Generate primes.
    # 2. For p in primes:
    #    - Generate all p-smooth numbers <= TARGET.
    #    - Put them in a set S.
    #    - Check if any a in S has TARGET-a in S.
    #    - If yes, return that pair.
    
    # This is much faster than factoring every number up to TARGET.
    # The number of p-smooth numbers grows slowly.
    # For p=23, how many 23-smooth numbers are there <= 123457?
    # Not too many.
    
    def is_smooth(n, primes_list):
        for p in primes_list:
            while n % p == 0:
                n //= p
            if n == 1:
                return True
            if p * p > n:
                break
        return n == 1

    # Better: Generate the set of smooth numbers directly.
    
    def gen_smooth_numbers(max_val, primes_list):
        smooth = {1}
        # Use a list to iterate, adding new numbers
        # Since primes_list is small, we can just do a BFS/DFS or iterative expansion
        # But simple iterative expansion:
        # Start with [1]
        # For each prime p in primes_list:
        #   For each existing number x:
        #     Add x*p, x*p^2, ... <= max_val
        # This can be done in-place if careful, or with a new set.
        
        # Let's use a set and expand it.
        # To avoid infinite loops, we process primes one by one.
        # Actually, the standard way is:
        # S = {1}
        # for p in primes:
        #   new_S = set()
        #   for x in S:
        #     y = x
        #     while y * p <= max_val:
        #       y *= p
        #       new_S.add(y)
        #   S.update(new_S)
        #   # Note: this misses combinations like 2*3, 2*3*5 etc if we are not careful?
        #   # No, because S already contains 2, 3, 2*3 from previous steps?
        #   # Wait.
        #   # Step 1: p=2. S={1, 2, 4, 8...}
        #   # Step 2: p=3.
        #   #   x=1 -> 3, 9, 27...
        #   #   x=2 -> 6, 18...
        #   #   x=4 -> 12, 36...
        #   #   So yes, it works.
        
        S = {1}
        for p in primes_list:
            new_nums = set()
            for x in S:
                y = x
                while y * p <= max_val:
                    y *= p
                    new_nums.add(y)
            S.update(new_nums)
        
        return S

    # We iterate p from smallest prime upwards.
    # current_primes = []
    
    for i, p in enumerate(primes):
        if time.time() > deadline:
            # Fallback
            return [1, TARGET - 1]
            
        current_primes = primes[:i+1]
        
        # Generate smooth numbers
        # Optimization: We only need to check if a pair sums to TARGET.
        # We can generate the set S.
        S = gen_smooth_numbers(TARGET, current_primes)
        
        # Check for pair
        for a in S:
            if a > TARGET:
                continue
            b = TARGET - a
            if b < 1:
                continue
            if b in S:
                # Found a pair with max prime factor <= p
                # Since we iterate p in increasing order, this is the minimum.
                return [a, b]
        
        # If we reach here, no solution with primes up to p.
        # Continue to next prime.
        
    # If no solution found in primes up to 100 (unlikely), fallback
    # We know p=23 works.
    return [1, TARGET - 1]
