"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'gaps': [[10007, 10009], [10009, 10037], [10037, 10039], [10039, 10061], [10061, 10067], [10067, 10069], [10069, 10079], [10079, 10091], [10091, 10093], [10093, 10099], [10099, 10103], [10103, 10111], [10111, 10133], [10133, 10139], [10139, 10141], [10141, 10151], [10151, 10159], [10159, 10163], [10163, 10169], [10169, 10177], [10177, 10181], [10181, 10193], [10193, 10211], [10211, 10223], [10223, 10243], [10243, 10247], [10247, 10253], [10253, 10259], [10259, 10267], [10267, 10271], [10271, 10273], [10273, 10289]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds margin
    
    # Find max gap to determine sieve bound
    max_gap = max(hi - lo for lo, hi in P['gaps'])
    # We need primes less than max_gap
    sieve_bound = max_gap + 10
    
    # Sieve for primes up to sieve_bound
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]
    
    primes = sieve(sieve_bound)
    
    # Determine the range of numbers we might need to check
    min_lo = min(lo for lo, hi in P['gaps'])
    max_hi = max(hi for lo, hi in P['gaps'])
    
    # For each gap, we need numbers m in (lo, hi) where lp(m) >= gap
    # This means m is not divisible by any prime < gap
    # Precompute for each possible gap size which primes to exclude
    
    # Strategy: For each gap, find the smallest m in (lo, hi) not divisible by any prime < gap
    # Use trial division with early termination
    
    result = [None] * len(P['gaps'])
    
    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break
            
        gap = hi - lo
        
        # Get primes less than gap
        relevant_primes = [p for p in primes if p < gap]
        
        witness = None
        
        # Try CRT: find m ≡ 1 (mod p) for all p < gap
        # m = k*M + 1 where M = product of all primes < gap
        if relevant_primes:
            M = 1
            for p in relevant_primes:
                M *= p
            
            # Find k such that lo < k*M + 1 < hi
            # lo - 1 < k*M < hi - 1
            # (lo - 1) / M < k < (hi - 1) / M
            k_min = (lo - 1) // M + 1
            k_max = (hi - 2) // M
            
            if k_min <= k_max:
                # Try all k in range
                for k in range(k_min, k_max + 1):
                    m = k * M + 1
                    if lo < m < hi:
                        # Verify: m should not be divisible by any p < gap
                        # By construction m ≡ 1 (mod p) for all p in relevant_primes
                        # So m % p == 1 for all p, which means m % p != 0
                        # But let's verify to be safe
                        valid = True
                        for p in relevant_primes:
                            if m % p == 0:
                                valid = False
                                break
                        if valid:
                            witness = m
                            break
        
        # If CRT didn't work, do sequential search
        if witness is None:
            for m in range(lo + 1, hi):
                # Check if m is divisible by any prime < gap
                is_valid = True
                for p in relevant_primes:
                    if m % p == 0:
                        is_valid = False
                        break
                    # Early termination: if p*p > m, we can stop
                    # (but m is at least lo+1 ~ 10008, and primes go up to ~30)
                    # So this optimization is minor
                if is_valid:
                    witness = m
                    break
        
        result[i] = int(witness) if witness is not None else None
    
    return result

# EVOLVE-BLOCK-END
