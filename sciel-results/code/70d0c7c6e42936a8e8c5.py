"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    """
    Find n in [1000, 1000000] that minimizes the number of distinct
    prime factors (from primes < 24) present in the "24-smooth parts"
    of n+1, n+2, ..., n+24.
    
    Primes strictly below 24: 2, 3, 5, 7, 11, 13, 17, 19, 23.
    There are 9 such primes.
    
    For each number m in the window, we compute its "smooth part"
    (the largest divisor of m composed only of these 9 primes).
    We want to minimize the count of distinct prime factors appearing
    across all 24 smooth parts.
    
    Strategy:
    1. Precompute for every number in [1001, 1000024] its smooth-part
       prime factor set (as a bitmask over the 9 primes).
    2. Slide a window of size 24 and count distinct primes in the
       union of the 24 bitmasks.
    3. Return the n that gives the minimum distinct count.
    
    Use a deadline guard to stay within 60s.
    """
    deadline = time.time() + 50  # safety margin before 60s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)  # 9
    
    # Max value we need: n + 24 where n up to 1000000
    max_val = 1000000 + 24
    
    # For each number, compute which of the 9 primes divide it (in its smooth part).
    # Actually, we need the smooth part's prime set. The smooth part of m is the
    # product of p^a for each prime p < 24 where p^a || m (i.e., the full power
    # of p dividing m). The set of primes in the smooth part is exactly the set
    # of primes p < 24 that divide m.
    #
    # So for each m, smooth_prime_mask[m] = bitmask of which of the 9 primes divide m.
    #
    # For a window [n+1, ..., n+24], the union of smooth prime masks gives the
    # set of distinct primes. We want to minimize popcount of this union.
    
    # Build the mask array using a sieve-like approach
    # mask[m] = 0 initially; for each prime p_i, set bit i for all multiples of p_i.
    mask = [0] * (max_val + 1)
    
    for idx, p in enumerate(primes):
        bit = 1 << idx
        # Set the bit for all multiples of p
        for m in range(p, max_val + 1, p):
            mask[m] |= bit
    
    # Now slide window of size 24 over n from 1000 to 1000000.
    # For window starting at n, the numbers are n+1, ..., n+24.
    # We need the OR of mask[n+1] | mask[n+2] | ... | mask[n+24].
    # Count bits in that OR.
    
    best_n = 1000
    best_count = 9  # max possible is 9
    
    # Use a running window OR with a count array for efficiency.
    # Since the window is small (24), we can just OR 24 values each step.
    # But that's 24 * 999000 = ~24M operations, which should be fine.
    
    # More efficient: use a frequency counter for each bit position.
    # freq[j] = number of numbers in current window whose mask has bit j set.
    # When a number enters/leaves the window, update freq.
    # The distinct count is the number of j with freq[j] > 0.
    
    freq = [0] * num_primes
    distinct = 0
    
    # Initialize window for n=1000: numbers 1001..1024
    for i in range(1001, 1025):
        m = mask[i]
        for j in range(num_primes):
            if m & (1 << j):
                freq[j] += 1
                if freq[j] == 1:
                    distinct += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        # Check deadline
        if time.time() > deadline:
            break
        
        # Remove n (which was the first element of previous window: n)
        # Previous window was [n, n+1, ..., n+23] for n-1? 
        # Wait, let me re-index.
        # 
        # For n=1000, window is [1001, 1024].
        # For n=1001, window is [1002, 1025].
        # So when going from n to n+1, we remove n+1 and add n+25.
        # 
        # Let me redo: for current n, window is [n+1, n+24].
        # Next n is n+1, window is [n+2, n+25].
        # Remove: n+1, Add: n+25.
        
        # Remove element n+1
        m_out = mask[n + 1]
        for j in range(num_primes):
            if m_out & (1 << j):
                freq[j] -= 1
                if freq[j] == 0:
                    distinct -= 1
        
        # Add element n+25
        m_in = mask[n + 25]
        for j in range(num_primes):
            if m_in & (1 << j):
                freq[j] += 1
                if freq[j] == 1:
                    distinct += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
