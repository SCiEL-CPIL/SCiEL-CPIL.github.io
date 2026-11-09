"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute SPF up to N_MAX + 1
    max_n = N_MAX + 1
    spf = list(range(max_n + 1))
    for i in range(2, int(max_n**0.5) + 1):
        if spf[i] == i:
            for j in range(i*i, max_n + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def _lpf_spf(n):
        if n <= 1:
            return 1
        lp = 1
        while n > 1:
            p = spf[n]
            lp = p
            while n % p == 0:
                n //= p
        return lp
    
    def _score_pair(n):
        return max(_lpf_spf(n), _lpf_spf(n + 1))

    # Strategy 1: Search for 11-smooth consecutive pairs
    # Generate all 11-smooth numbers in range
    def generate_smooth_numbers(primes, max_val):
        nums = set([1])
        for p in primes:
            new_nums = set()
            for n in nums:
                v = n
                while v * p <= max_val:
                    v *= p
                    new_nums.add(v)
            nums.update(new_nums)
        return [n for n in nums if N_MIN <= n <= N_MAX]

    # Check 11-smooth
    primes_11 = [2, 3, 5, 7, 11]
    smooth_11 = generate_smooth_numbers(primes_11, N_MAX)
    smooth_11_set = set(smooth_11)
    
    best_n = N_MIN
    best_val = _score_pair(N_MIN)
    
    found_11 = False
    for n in smooth_11:
        if (n + 1) in smooth_11_set:
            # Check if n+1 is in range
            if n + 1 <= N_MAX:
                val = _score_pair(n)
                if val < best_val:
                    best_val = val
                    best_n = n
                found_11 = True
                # We can break if we find one, but maybe there's a smaller L(n)?
                # L(n) will be <= 11. Any 11-smooth pair has L(n) <= 11.
                # We want to minimize L(n).
                # If we find one, L(n) is at most 11.
                # We should check all to find the min L(n) among them.
                # But L(n) is determined by the largest prime factor.
                # For 11-smooth numbers, L(n) <= 11.
                # It could be 2, 3, 5, 7, or 11.
                # We want the smallest L(n).
                # So we should check all pairs and pick the one with min L(n).
                pass

    if found_11:
        # Re-scan the found pairs to find min L(n)
        min_val = float('inf')
        min_n = N_MIN
        for n in smooth_11:
            if (n + 1) in smooth_11_set and n + 1 <= N_MAX:
                val = _score_pair(n)
                if val < min_val:
                    min_val = val
                    min_n = n
        best_val = min_val
        best_n = min_n

    # If no 11-smooth pair found, or even if found, do a full scan to be sure?
    # The full scan is fast. Let's just do the full scan if we haven't found L(n) < 11?
    # Actually, if we found an 11-smooth pair, L(n) <= 11.
    # The linear scan will find the global minimum.
    # So we can just run the linear scan.
    # But the linear scan might be slow? No, it's 1M iterations.
    
    # To be safe and fast, if we found an 11-smooth pair, we can just return the best one.
    # But what if there is a 7-smooth pair?
    # The linear scan will find it.
    
    # Let's just run the linear scan if time permits.
    # Since we have 85s, and 1M scan takes < 1s, we can always do it.
    
    # Update best with linear scan
    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline - 1.0:
            break
        current_val = _score_pair(n)
        if current_val < best_val:
            best_val = current_val
            best_n = n

    return best_n

# EVOLVE-BLOCK-END
