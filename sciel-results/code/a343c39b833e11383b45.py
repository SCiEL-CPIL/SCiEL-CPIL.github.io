import time

def construct():
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 55  # Safety margin
    
    # Sieve for primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    # We need primes up to some limit. 
    # If the optimal max prime factor is P*, we need to test primes up to P*.
    # P* is likely small. Let's try primes up to 1000.
    primes = sieve(1000)
    
    # We will iterate through primes in increasing order.
    # For each prime p, we generate p-smooth numbers.
    
    for idx, p in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
            
        # Generate p-smooth numbers <= TARGET
        # We use the primes primes[0..idx]
        smooths = set()
        
        def dfs(val, start_idx):
            if val > TARGET:
                return
            smooths.add(val)
            for i in range(start_idx, idx + 1):
                q = primes[i]
                next_val = val * q
                if next_val > TARGET:
                    break
                dfs(next_val, i)
                
        dfs(1, 0)
        
        # Check for pair
        for a in smooths:
            b = TARGET - a
            if b >= 1 and b in smooths:
                # Found a pair with max prime factor <= p
                # Is it optimal? 
                # Since we iterate p in increasing order, this is the first time
                # a valid pair appears. 
                # However, it's possible that a pair with a smaller max prime factor 
                # exists but wasn't in the set for a smaller p? 
                # No, if max(P(a), P(b)) = Q, then a and b are Q-smooth.
                # So they would have been found when we tested p=Q.
                # Thus, the first p for which we find a pair gives the minimal 
                # possible upper bound on the max prime factor.
                # The actual max prime factor of this specific pair might be 
                # smaller than p, but it cannot be smaller than the p of the 
                # previous iteration (where no pair was found).
                # So this is the optimal solution.
                return [a, b]
                
    # Fallback if no pair found within prime limit
    # This is unlikely for S=123457
    # Just return 1 and TARGET-1? Or do a quick search.
    # 1 is 1-smooth (P(1)=1). TARGET-1 = 123456.
    # 123456 = 2^4 * 3 * 11 * 29 * 19? Let's check.
    # 123456 / 2 = 61728 / 2 = 30864 / 2 = 15432 / 2 = 7716 / 2 = 3858 / 2 = 1929.
    # 1929 / 3 = 643. 643 is prime? sqrt(643) ~ 25. 
    # 643 not div by 2,3,5,7 (643=7*91+6), 11 (643=11*58+5), 13 (643=13*49+6), 17 (643=17*37+14), 19 (643=19*33+16), 23 (643=23*27+22).
    # So P(123456) = 643.
    # So [1, 123456] has max prime factor 643.
    # Our algorithm should find a better pair.
    
    # Simple fallback
    return [1, TARGET - 1]
