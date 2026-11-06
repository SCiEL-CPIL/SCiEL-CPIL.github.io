import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s kill (or 150s as per prompt? "KILLS after 150s", "finish within 120s". I will use 110s to be safe)
    
    # 1. Generate primes up to 10^6
    # sqrt(10^12) = 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    def phi(n):
        if n == 1:
            return 1
        result = n
        y = n
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                while y % p == 0:
                    y //= p
                result -= result // p
        if y > 1:
            result -= result // y
        return result

    best_start = 1
    best_len = 1
    
    # We want to find a run of length 200.
    # Let's try some specific large starts where collisions are unlikely.
    # Also try some smaller ones just in case.
    
    candidates = [
        10**12 - 199,
        10**12 - 1000,
        10**12 - 10000,
        10**11,
        10**11 + 12345,
        10**10,
        10**9,
        10**8,
        10**7,
        10**6
    ]
    
    # Add some random candidates in high range
    for _ in range(20):
        candidates.append(random.randint(10**11, 10**12 - 200))
    
    for s in candidates:
        if time.time() > DEADLINE:
            break
        if s < 1 or s + 199 > 10**12:
            continue
            
        # Check window [s, s+199]
        vals = []
        for i in range(200):
            if time.time() > DEADLINE:
                break
            vals.append(phi(s + i))
        
        if len(vals) < 200:
            continue
            
        # Check for duplicates
        seen = set()
        current_run_start = 0
        max_run_in_window = 0
        current_run_len = 0
        
        # We need pairwise distinct.
        # If we find a duplicate, the run breaks.
        # We want the longest contiguous subsequence with distinct values.
        
        # Reset seen for the whole window check?
        # No, we just need to check if the whole window of 200 is valid.
        # If not, we can find the max sub-run.
        
        # Simple check for full 200:
        if len(set(vals)) == 200:
            return {'start': s, 'length': 200}
            
        # If not full 200, find max sub-run
        # This is O(k^2) if done naively, but k=200 is small.
        # We can slide a window or just check all sub-windows ending at each point?
        # Actually, we can just iterate and keep a set.
        # When a duplicate is found, we need to remove elements from the start until the duplicate is gone?
        # Yes, sliding window approach.
        
        left = 0
        seen = {}
        for right in range(200):
            v = vals[right]
            if v in seen:
                # Move left until the previous occurrence of v is removed
                while left <= right and vals[left] != v:
                    seen.pop(vals[left], None)
                    left += 1
                if left <= right and vals[left] == v:
                    seen.pop(v, None)
                    left += 1
            seen[v] = right
            current_len = right - left + 1
            if current_len > max_run_in_window:
                max_run_in_window = current_len
                if max_run_in_window > best_len:
                    best_len = max_run_in_window
                    best_start = s + left
        
    return {'start': best_start, 'length': best_len}
