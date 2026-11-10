"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small_factor using efficient sieve-like approach
    # small_factor[x] = product of all prime factors < 24 (with multiplicity)
    # Use a more efficient method: for each prime, divide out all powers at once
    small_factor = [1] * (max_val + 1)
    small_factor[0] = 0
    
    for p in small_primes:
        p_power = p
        # For each power p^k, multiply p into all multiples of p^k
        # This is equivalent to: for each multiple of p, multiply p^(v_p(multiple))
        # More efficient: for each multiple m of p, count how many times p divides m
        for multiple in range(p, max_val + 1, p):
            val = multiple
            while val % p == 0:
                small_factor[multiple] *= p
                val //= p
    
    # Sliding window approach with array-based frequency counting
    # For candidate n, numbers are n+1 .. n+24
    
    best_score = 25
    best_n = 1000
    
    # Use an array for frequency counting
    # The maximum possible small_factor value is bounded
    # For numbers up to 1000024, the max small-prime factor is at most
    # 2^19 * 3^12 * 5^7 * 7^5 * 11^3 * 13^3 * 17^3 * 19^3 * 23^3
    # but in practice it's much smaller. We use a dict-like approach with array.
    # Actually, let's just use a dictionary since the number of distinct values is small.
    
    freq = {}
    distinct = 0
    
    # Initialize window for n=1000: factors of 1001..1024
    for i in range(1, 25):
        f = small_factor[1000 + i]
        if f not in freq:
            freq[f] = 1
            distinct += 1
        else:
            freq[f] += 1
    
    if distinct < best_score:
        best_score = distinct
        best_n = 1000
    
    # Slide window: for n from 1001 to 1000000
    # Remove n, add n+24
    # Check time every 10000 iterations to reduce overhead
    check_interval = 10000
    last_check = start_time
    
    for n in range(1001, 1000001):
        # Remove element at position n
        remove_f = small_factor[n]
        freq[remove_f] -= 1
        if freq[remove_f] == 0:
            del freq[remove_f]
            distinct -= 1
        
        # Add element at position n+24
        add_f = small_factor[n + 24]
        if add_f not in freq:
            freq[add_f] = 1
            distinct += 1
        else:
            freq[add_f] += 1
        
        if distinct < best_score:
            best_score = distinct
            best_n = n
            if best_score == 1:
                break
        
        # Periodic time check
        if n % check_interval == 0:
            if time.time() > deadline:
                break
    
    return best_n
# EVOLVE-BLOCK-END
