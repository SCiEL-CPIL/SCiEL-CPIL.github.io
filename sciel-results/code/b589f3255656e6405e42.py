import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    N = 1000000
    LOW = 10000
    LIMIT = N + 1  # We need to check n+1 up to 1000001
    
    # Sieve of Eratosthenes to get primes up to LIMIT
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            for j in range(i * i, LIMIT + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, LIMIT + 1) if is_prime[i]]
    
    # Set of smooth numbers
    smooth_set = {1}
    
    # We will iterate through primes in increasing order
    # For each prime p, we add p to the set of generators
    # and generate new smooth numbers
    
    best_n = None
    best_val = float('inf')
    
    for p in primes:
        if time.time() > deadline:
            break
            
        # Generate new smooth numbers using p
        # We iterate over a copy of the current smooth_set to avoid modification during iteration
        current_smooth = list(smooth_set)
        
        for s in current_smooth:
            val = s
            while val * p <= LIMIT:
                val *= p
                smooth_set.add(val)
                
        # Check for adjacent pairs
        # We need n in [LOW, N] such that n in smooth_set and n+1 in smooth_set
        # To optimize, we can check only the new numbers added?
        # No, because n+1 might be old and n might be new, or vice versa.
        # But if a pair existed before, we would have returned it.
        # So we only need to check pairs involving the NEW numbers.
        
        # Actually, it's simpler to just check the whole set if it's small, 
        # or check new numbers against the set.
        
        # Let's check new numbers
        # We don't track "new" explicitly, but we can assume that if a pair (n, n+1) 
        # was valid for previous primes, we would have found it.
        # So any valid pair found now must involve at least one number that became 
        # smooth due to the inclusion of p.
        
        # However, implementing "new" tracking is complex.
        # Given the small size of smooth_set for small p, 
        # we can just iterate through the sorted smooth_set?
        # Or just check the new numbers added in this step.
        
        # Let's just check if any n in [LOW, N] is in smooth_set and n+1 is in smooth_set.
        # This is O(1) per check if we use a set, but we need to iterate n?
        # No, we can iterate through the smaller of the two sets?
        # Or just iterate through the new numbers added.
        
        # Let's collect new numbers
        # Actually, the loop above adds to smooth_set.
        # We can just check the entire smooth_set for pairs?
        # If |smooth_set| is small, it's fast.
        
        # Optimization: Only check n such that n is in smooth_set.
        # We can iterate through smooth_set, filter n in [LOW, N], and check n+1.
        
        found = False
        # To avoid O(|S|) check every time, we can be smarter.
        # But for now, let's just do it.
        
        # Sort is not needed for set membership.
        
        # We can break early if we find a valid n.
        # But we want the MINIMAL p.
        # So the first p for which a pair exists is the optimal one.
        
        # So we just need to check if ANY pair exists.
        
        # Iterate through smooth_set
        # If it's too slow, we can optimize.
        
        for n in smooth_set:
            if n < LOW or n > N:
                continue
            if (n + 1) in smooth_set:
                return n
        
        # If we reach here, no pair found with current p.
        # Continue to next prime.

    # If loop finishes without returning, fallback
    # This should not happen as for p >= LIMIT, all numbers are smooth.
    # But to be safe, return a heuristic or the best found.
    
    # Fallback: Simple scan (slow, but correct)
    # Or return LOW
    return LOW
