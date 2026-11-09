import time
import random
from itertools import combinations

# Precompute smallest prime factor for numbers up to 20000 (max sum = 2*10000)
MAX_SUM = 20000
spf = list(range(MAX_SUM + 1))
for i in range(2, int(MAX_SUM**0.5) + 1):
    if spf[i] == i:
        for j in range(i*i, MAX_SUM + 1, i):
            if spf[j] == j:
                spf[j] = i

def prime_factors(n):
    """Return set of distinct prime factors of n."""
    if n <= 1:
        return set()
    factors = set()
    while n > 1:
        p = spf[n]
        factors.add(p)
        while n % p == 0:
            n //= p
    return factors

def count_distinct_primes(arr):
    """Count distinct prime factors across all pairwise sums of distinct elements."""
    all_primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i+1, n):
            s = arr[i] + arr[j]
            all_primes |= prime_factors(s)
    return len(all_primes)

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    best_score = 999
    best_arr = list(range(1, 13))  # fallback
    
    # Strategy 1: Powers of 2 (structured family)
    # 2^0, 2^1, ..., 2^11 -> 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048
    # All <= 10000, distinct
    powers_of_2 = [1 << i for i in range(12)]
    if max(powers_of_2) <= 10000:
        score = count_distinct_primes(powers_of_2)
        if score < best_score:
            best_score = score
            best_arr = powers_of_2[:]
    
    # Strategy 2: Try different bases and offsets
    # For a base b, consider {b^0, b^1, ..., b^11} if max <= 10000
    for base in range(2, 11):
        if base**11 > 10000:
            continue
        arr = [base**i for i in range(12)]
        if len(set(arr)) == 12 and max(arr) <= 10000:
            score = count_distinct_primes(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
        if time.time() > deadline:
            return best_arr
    
    # Strategy 3: Greedy construction with small prime set
    # Try to find 12 numbers where all pairwise sums have few prime factors
    # Start with numbers that are all even (so sums are even, at least factor 2)
    # Then try to minimize other prime factors
    
    # Approach: Pick 12 numbers from a set where pairwise sums are "smooth"
    # Try multiples of a small number
    for mult in range(1, 20):
        if time.time() > deadline:
            return best_arr
        # Try numbers of form mult * k for k = 1..12, 13..24, etc.
        for start_k in range(1, 30):
            if time.time() > deadline:
                return best_arr
            arr = [mult * (start_k + i) for i in range(12)]
            if len(set(arr)) < 12 or max(arr) > 10000:
                continue
            score = count_distinct_primes(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
    
    # Strategy 4: Local search / hill climbing
    # Start from best found, try swapping elements
    if best_score > 5:  # Only if we haven't found something very good
        random.seed(42)
        current_arr = best_arr[:]
        current_score = best_score
        for _ in range(200):
            if time.time() > deadline:
                return best_arr
            # Try a random perturbation: replace one element with a random value
            idx = random.randint(0, 11)
            old_val = current_arr[idx]
            new_val = random.randint(1, 10000)
            if new_val in current_arr and new_val != old_val:
                continue
            current_arr[idx] = new_val
            score = count_distinct_primes(current_arr)
            if score < current_score:
                current_score = score
                if score < best_score:
                    best_score = score
                    best_arr = current_arr[:]
            else:
                current_arr[idx] = old_val
    
    # Strategy 5: Try structured sets based on powers of 2 with offsets
    # e.g., {c * 2^i for i in 0..11}
    for c in range(1, 20):
        if time.time() > deadline:
            return best_arr
        arr = [c * (1 << i) for i in range(12)]
        if len(set(arr)) == 12 and max(arr) <= 10000:
            score = count_distinct_primes(arr)
            if score < best_score:
                best_score = score
                best_arr = arr[:]
    
    return best_arr
