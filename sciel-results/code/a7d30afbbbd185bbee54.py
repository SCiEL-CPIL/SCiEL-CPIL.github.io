import time
import numpy as np

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_smooth_numbers(primes, limit):
    nums = [1]
    def dfs(idx, curr):
        if idx == len(primes):
            if curr <= limit:
                nums.append(curr)
            return
        p = primes[idx]
        power = 1
        while curr * power <= limit:
            dfs(idx + 1, curr * power)
            power *= p
    dfs(0, 1)
    return sorted(nums)

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    max_val = 1000000
    min_n = 10000
    
    # Step 1: Check for very smooth consecutive pairs (LPF <= 11, 7, 5)
    # This is a targeted search for a potentially better (lower) score.
    # If found, we return immediately.
    for y in [11, 7, 5]:
        primes = [p for p in range(2, y + 1) if is_prime(p)]
        smooths = generate_smooth_numbers(primes, max_val)
        
        # Find consecutive pairs in the sorted list
        # We need n >= 10000
        # Iterate through the sorted list
        for i in range(len(smooths) - 1):
            if smooths[i] < min_n:
                continue
            if smooths[i + 1] - smooths[i] == 1:
                # Found a pair (n, n+1) where both are y-smooth
                # Max LPF is <= y
                return int(smooths[i])
        
        # Time check
        if time.time() > deadline:
            break

    # Step 2: Fallback to Sieve for Largest Prime Factor
    # This finds the global minimum if no very smooth pair exists.
    # The current best known is 13.
    
    lpf = np.zeros(max_val + 1, dtype=np.int32)
    
    for p in range(2, max_val + 1):
        if lpf[p] == 0:
            lpf[p::p] = p
            
        if p % 100000 == 0:
            if time.time() > deadline:
                break

    if time.time() > deadline:
        return 10000

    # Vectorized calculation of scores
    # n ranges from 10000 to 1000000
    # We want min of max(lpf[n], lpf[n+1])
    scores = np.maximum(lpf[min_n:max_val + 1], lpf[min_n + 1:max_val + 2])
    
    if len(scores) == 0:
        return 10000
        
    min_score_idx = np.argmin(scores)
    best_n = min_n + min_score_idx
    
    return int(best_n)
