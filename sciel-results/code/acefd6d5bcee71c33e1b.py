"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Max n is 1000000, so we need factors for numbers up to 1000024
    max_val = 1000024
    
    # Precompute small-prime factor for each number
    # small_factor[i] = product of p^e for all p < 24 dividing i
    small_factor = list(range(max_val + 1))

    # Better approach: initialize to 1, then multiply by p^e for each small prime
    small_factor = [1] * (max_val + 1)
    
    for p in small_primes:
        # For each power p^k, mark multiples
        power = p
        while power <= max_val:
            for multiple in range(power, max_val + 1, power):
                # Check if p^k actually divides multiple (i.e., p^(k+1) does not)
                # Actually, we want to multiply by p for each factor of p
                # Simpler: for each multiple of p, divide out p and multiply small_factor by p
                # But that's O(n log n). Let's use a different approach.
                pass
            power *= p
    
    # Even better: for each number, compute its small-prime factor directly
    # Using a sieve-like approach
    small_factor = [1] * (max_val + 1)
    
    for p in small_primes:
        for multiple in range(p, max_val + 1, p):
            # Count how many times p divides multiple
            val = multiple
            while val % p == 0:
                small_factor[multiple] *= p
                val //= p
    
    # Now slide a window of 24 and count distinct small factors
    best_count = 25  # max possible is 9 distinct primes, but let's be safe
    best_n = 1000
    
    # For efficiency, use a frequency array or set
    # Since small factors can be large, use a dict for counting
    from collections import Counter
    
    # Initial window: n=1000, numbers 1001 to 1024
    window_factors = [small_factor[i] for i in range(1001, 1025)]
    freq = Counter(window_factors)
    current_distinct = len(freq)
    
    if current_distinct < best_count:
        best_count = current_distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Remove n (which was n+0 in previous window, now leaving)
        # Previous window was (n-1)+1 to (n-1)+24 = n to n+23
        # New window is n+1 to n+24
        # So remove small_factor[n], add small_factor[n+24]
        
        old_val = small_factor[n]
        new_val = small_factor[n + 24]
        
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        freq[new_val] = freq.get(new_val, 0) + 1
        
        current_distinct = len(freq)
        
        if current_distinct < best_count:
            best_count = current_distinct
            best_n = n
        
        # Check time
        if time.time() > deadline:
            break
    
    return best_n
# EVOLVE-BLOCK-END
