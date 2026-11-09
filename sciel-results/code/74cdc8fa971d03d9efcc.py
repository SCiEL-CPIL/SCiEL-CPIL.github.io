"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start = time.time()
    deadline = start + 85  # safety margin before 90s limit

    N = 1000000  # max n
    WINDOW = 24
    PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    MAX_VAL = N + WINDOW

    # Compute sp[m] = complete factor of m composed of primes < 24
    # Use NumPy for speed
    sp = np.ones(MAX_VAL + 1, dtype=np.int64)
    
    for p in PRIMES:
        # For each prime p, divide out all powers of p
        # sp[m] should include p^k where p^k divides m
        # We can do this by: for m divisible by p, sp[m] *= p, then m //= p
        # But we need to do this iteratively for all powers
        
        # Better approach: compute the p-part of each number
        # For m in range(1, MAX_VAL+1):
        #   while m % p == 0: sp[m] *= p; m //= p
        # But this is O(MAX_VAL * log(MAX_VAL)) per prime
        
        # Alternative: use the fact that sp[m] = product of p^v_p(m) for p in PRIMES
        # We can compute this using a sieve-like approach
        
        # For each prime p, compute p_part[m] = p^v_p(m)
        # Then sp[m] = product of p_part[m] for all p in PRIMES
        
        # Efficient way: for each p, mark multiples and multiply
        p_part = np.ones(MAX_VAL + 1, dtype=np.int64)
        for m in range(p, MAX_VAL + 1, p):
            # Count how many times p divides m
            val = m
            count = 0
            while val % p == 0:
                count += 1
                val //= p
            p_part[m] = p ** count
        
        sp *= p_part

    # Now sp[m] is the complete factor composed of primes < 24
    # Find n in [1000, N] minimizing distinct count of sp[n+1..n+24]
    
    best_n = 1000
    best_distinct = float('inf')
    
    # Use a sliding window approach with a frequency counter
    # For efficiency, process in chunks
    
    # First, let's try a random sampling approach combined with local search
    # Sample many random n values, then do local search around the best ones
    
    np.random.seed(42)
    
    # Sample 100000 random n values
    sample_size = 100000
    samples = np.random.randint(1000, N + 1, size=sample_size)
    
    best_sample_n = 1000
    best_sample_distinct = float('inf')
    
    for idx in range(0, sample_size, 1000):
        if time.time() > deadline - 10:
            break
        batch = samples[idx:idx+1000]
        for n in batch:
            vals = sp[n+1:n+WINDOW+1]
            distinct = len(np.unique(vals))
            if distinct < best_sample_distinct:
                best_sample_distinct = distinct
                best_sample_n = n
    
    # Local search around the best sample
    if time.time() < deadline - 5:
        center = best_sample_n
        search_radius = 5000
        for n in range(max(1000, center - search_radius), min(N + 1, center + search_radius + 1)):
            if time.time() > deadline - 3:
                break
            vals = sp[n+1:n+WINDOW+1]
            distinct = len(np.unique(vals))
            if distinct < best_distinct:
                best_distinct = distinct
                best_n = n
    
    # Also do a broader scan in chunks if time permits
    if time.time() < deadline - 10:
        chunk_size = 50000
        for start_n in range(1000, N + 1, chunk_size):
            if time.time() > deadline - 5:
                break
            end_n = min(start_n + chunk_size, N + 1)
            for n in range(start_n, end_n):
                vals = sp[n+1:n+WINDOW+1]
                distinct = len(np.unique(vals))
                if distinct < best_distinct:
                    best_distinct = distinct
                    best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
