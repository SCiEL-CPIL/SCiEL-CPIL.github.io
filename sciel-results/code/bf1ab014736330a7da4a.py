import time
import math
import random

def _gen_primes_upto(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    if limit < 2:
        return []
    # Use bytearray for memory efficiency and speed
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            # Mark multiples of i as not prime
            start = i * i
            step = i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Precompute primes up to a sufficient limit
    # We need primes up to the maximum p we will test.
    # Let's test p up to 5,000,000 first.
    # The segmented sieve for a candidate p needs primes up to p.
    # So we need a list of primes up to 5,000,000.
    PRIME_LIMIT = 5_000_000
    primes = _gen_primes_upto(PRIME_LIMIT)
    
    # We will iterate through these primes as candidates for p.
    best_k = 0
    best_p = 2
    
    # To speed up the check, we can use a helper function
    def is_p_smooth_run(p, k_max=200):
        """
        Checks if p^2+1, ..., p^2+k_max are p-smooth.
        Returns the maximum k such that p^2+1...p^2+k are p-smooth.
        Uses segmented sieve logic: divide out all primes q <= p from the interval.
        """
        base = p * p
        # We need to check numbers base+1 to base+k_max
        # Initialize the remainders
        rem = [base + i for i in range(1, k_max + 1)]
        
        # Iterate through all primes q <= p
        # We can stop if q*q > base + k_max? No, we need to remove ALL factors <= p.
        # If a number has a factor > p, it's not p-smooth.
        # So we must remove all factors <= p.
        # If after removing all factors <= p, the remainder is 1, it's p-smooth.
        # If remainder > 1, it has a prime factor > p (since all smaller ones removed), so it's not p-smooth.
        
        # Optimization: We only need to iterate primes q <= p.
        # But we can break early if q > p.
        # Also, if q > base + k_max, q cannot divide any number in the range.
        # But p is much smaller than base, so q <= p is the binding constraint.
        
        # Efficient iteration:
        # We have a list 'primes'. We need primes <= p.
        # We can find the index of the last prime <= p.
        # Since we are iterating p in increasing order from the same list, 
        # we can just iterate the primes list up to p.
        
        # However, doing this from scratch for every p is O(N^2) in number of primes.
        # Is there a faster way?
        # The cost is sum over p of (number of primes <= p).
        # This is roughly sum_{i=1}^{N} i = N^2/2.
        # N = 350,000 (primes up to 5M). N^2/2 is huge.
        
        # We need a better strategy.
        # Instead of checking every p exhaustively with a full sieve pass, 
        # we can use a different approach:
        # For a fixed p, the check is fast (~500 ops).
        # But doing it for 350,000 primes is 350,000 * 500 = 1.75 * 10^8 ops.
        # In Python, this might take 10-20 seconds. This is acceptable.
        
        # Let's implement the check carefully.
        
        # We need the list of primes <= p.
        # Since we are iterating p from the primes list, let's say p = primes[idx].
        # The primes <= p are primes[0...idx].
        
        # Pass idx to the function? Or just use the global list and break when q > p.
        
        for q in primes:
            if q > p:
                break
            # Find the first multiple of q in [base+1, base+k_max]
            # start = ((base + 1) // q) * q
            # If start <= base, start += q
            # Actually, we want start >= base + 1
            start = ((base + 1 + q - 1) // q) * q
            if start <= base:
                start += q
            
            # Divide q out of rem[j] for all j where rem[j] is divisible by q
            # rem[j] corresponds to base + 1 + j
            # The value at index j is base + 1 + j.
            # We need to find j such that (base + 1 + j) % q == 0.
            # This is equivalent to (base + 1) % q == -j % q.
            # Or simply iterate j from 0 to k_max-1 and check divisibility?
            # No, that's O(k_max) per prime.
            # Better to jump by q.
            
            # Index in rem array corresponding to value 'val' is val - (base + 1).
            # So for val = start, start+q, ...
            # idx = val - base - 1
            
            for val in range(start, base + k_max + 1, q):
                idx = val - base - 1
                if 0 <= idx < k_max:
                    while rem[idx] % q == 0:
                        rem[idx] //= q
                    # Early exit optimization:
                    # If rem[idx] becomes 1, it's fully factored.
                    # If rem[idx] > 1 and rem[idx] < q*q? No, q is the current prime.
                    # If rem[idx] > 1, it might have larger factors.
                    # We continue.
        
        # After processing all primes q <= p, check if all rem[i] are 1
        # If any rem[i] > 1, it has a prime factor > p, so it's not p-smooth.
        # The run length k is the number of initial 1s.
        
        k = 0
        for i in range(k_max):
            if rem[i] == 1:
                k += 1
            else:
                break
        return k

    # Iterate through candidate primes
    # We can't iterate ALL 350k primes if each takes 5ms (1750s).
    # Let's estimate the time for one check.
    # For p=5M, ~350k primes to check.
    # Inner loop: sum(200/q) for q <= p.
    # This is ~200 * log(log p) ~ 200 * 2.8 ~ 560 operations.
    # So one check is ~560 operations.
    # 350,000 checks * 560 ops = 196,000,000 ops.
    # Python can do ~10-50 million simple ops per second.
    # So 200M ops might take 4-20 seconds.
    # This is within the 110s limit.
    
    # Let's run it.
    for p in primes:
        if time.time() > deadline:
            break
        
        k = is_p_smooth_run(p)
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we have time, we can try larger primes?
    # The probability of long runs might be related to p.
    # But we've exhausted primes up to 5M.
    # Let's return the best found.
    
    return {'p': best_p, 'k': best_k}
