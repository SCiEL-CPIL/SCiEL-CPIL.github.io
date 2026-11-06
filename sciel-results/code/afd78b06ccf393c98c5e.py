"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # For each prime, compute its bitmask contribution
    # We'll use a bitmask where bit j is set if prime j divides the number
    
    # Precompute for each prime, the residues mod p that are 0
    # For a number m, prime p divides m iff m % p == 0
    
    # Strategy: For each n in [1000, 1000000], we have 24 numbers n+1..n+24
    # For each number, compute its small-prime factor mask (9-bit)
    # The "distinct extracted factors" = number of distinct primes that appear in ANY of the 24 masks
    # This is the bit-count of (mask_1 | mask_2 | ... | mask_24)
    # We want to MINIMIZE this count
    
    # Use sliding window with frequency array of size 512 (2^9)
    # freq[mask] = count of numbers in window with exactly that mask
    # To compute OR of all masks in window, we need to know which bits are present
    # bit_present[j] = number of masks in window that have bit j set
    # A bit j is in the OR iff bit_present[j] > 0
    
    # So we maintain bit_present[0..8] and the OR is the count of bit_present[j] > 0
    
    # Precompute mask for each residue class
    # For number m, mask = sum over j of (1 << j) if m % primes[j] == 0
    
    # For sliding window: when we slide from n to n+1:
    # - Remove number n+1 (its mask)
    # - Add number n+25 (its mask)
    
    # Let's precompute masks for a large range using the fact that
    # m % p depends only on m mod p. For multiple primes, mask depends on m mod LCM(primes)
    # LCM(2,3,5,7,11,13,17,19,23) = product since they're all prime = 2*3*5*7*11*13*17*19*23
    # = 9,699,690 which is too large to precompute fully
    
    # Alternative: compute mask on the fly for each number
    # mask(m) = 0
    # for j, p in enumerate(primes):
    #     if m % p == 0:
    #         mask(m) |= (1 << j)
    
    # This is 9 divisions per number. For a sliding window of 24, we need 2 new numbers per step
    # (one removed, one added). Actually we just need to compute mask for the new number added.
    
    # Total numbers to check: ~1,000,000 positions, each requiring 1 mask computation (9 mods)
    # That's about 9 million mod operations, which should be fast enough in Python if optimized.
    
    # Let's be smarter: precompute mask for numbers in a block using periodicity
    # Actually, let's just do it directly with optimization.
    
    # Precompute mask for all numbers from 1001 to 1000025 (n+25 where n up to 1000000)
    # That's about 1 million numbers, each needing 9 modulo checks
    
    # Better approach: for each prime p, precompute a boolean array of length p
    # is_div[j][r] = (r == 0) for prime j, residue r
    # Then mask(m) can be computed by looking up m % p for each p
    
    # Even better: use the Chinese Remainder Theorem periodicity
    # The mask pattern repeats every LCM(2,3,5,7,11,13,17,19,23) steps
    # But that LCM is ~9.7M, too large
    
    # Let's just compute masks incrementally. For the sliding window:
    # We need mask(n+1) through mask(n+24) initially, then slide.
    
    # Optimization: compute mask for a range of consecutive numbers efficiently
    # For each prime p, the numbers divisible by p in [a, b] are a//p*p, (a//p+1)*p, ...
    
    # Let me try a different approach: precompute the mask for each number in a reasonable range
    # using a sieve-like method
    
    max_n = 1000000
    max_val = max_n + 25  # we need masks up to n+24 where n <= max_n
    
    # Sieve-like: for each prime, mark which numbers in our range are divisible
    # Use a bytearray for each prime's divisibility, then combine
    
    # Actually, let's just compute the mask array directly
    # mask_arr[m] for m in [1001, max_val]
    
    # To save memory, we only need masks for the sliding window approach
    # But let's precompute all masks first since it's a one-time cost
    
    # Number of values: max_val - 1000 + 1 ≈ 1,000,026
    # Each mask is 9 bits, fits in a byte
    
    mask_arr = bytearray(max_val + 1)
    
    for j, p in enumerate(primes):
        bit = 1 << j
        # Mark all multiples of p in [1001, max_val]
        first = ((1001 + p - 1) // p) * p
        for m in range(first, max_val + 1, p):
            mask_arr[m] |= bit
    
    # Now use sliding window
    # Window is [n+1, n+24], so 24 consecutive numbers
    # bit_count[j] = number of numbers in window divisible by prime j
    # distinct_count = number of j where bit_count[j] > 0
    
    best_n = 1000
    best_score = 9  # max possible
    
    # Initialize window for n=1000: numbers 1001 to 1024
    bit_counts = [0] * num_primes
    for m in range(1001, 1025):
        mask = mask_arr[m]
        for j in range(num_primes):
            if mask & (1 << j):
                bit_counts[j] += 1
    
    current_distinct = sum(1 for c in bit_counts if c > 0)
    if current_distinct < best_score:
        best_score = current_distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, max_n + 1):
        # Check time
        if time.time() > deadline:
            break
        
        # Remove n (the number leaving the window)
        mask_out = mask_arr[n]
        for j in range(num_primes):
            if mask_out & (1 << j):
                bit_counts[j] -= 1
        
        # Add n+24 (the number entering the window)
        mask_in = mask_arr[n + 24]
        for j in range(num_primes):
            if mask_in & (1 << j):
                bit_counts[j] += 1
        
        current_distinct = sum(1 for c in bit_counts if c > 0)
        if current_distinct < best_score:
            best_score = current_distinct
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
