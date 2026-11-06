import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin

    # 1. Sieve primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]

    def phi_fast(n):
        if n == 1:
            return 1
        result = n
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        return result

    best_k = 1
    best_s = 1

    # 2. Baseline scan: 1 to 200,000
    # Using sliding window
    seen = {}
    window_start = 1
    for i in range(1, 200001):
        if time.time() > deadline - 20:
            break
        p_val = phi_fast(i)
        if p_val in seen:
            length = i - window_start
            if length > best_k:
                best_k = length
                best_s = window_start
            if seen[p_val] >= window_start:
                window_start = seen[p_val] + 1
        seen[p_val] = i
    
    length = 200000 - window_start + 1
    if length > best_k:
        best_k = length
        best_s = window_start

    # 3. Random sampling with local extension
    # We check a window of size 100 to be safe.
    # If we find a run, we extend it.
    
    def extend_run(s, k, vals):
        # vals is set of phi values in [s, s+k-1]
        # Try to extend left
        new_s = s
        new_k = k
        current_vals = set(vals)
        
        # Extend Left
        while new_s > 1:
            prev_val = phi_fast(new_s - 1)
            if prev_val in current_vals:
                break
            current_vals.add(prev_val)
            new_s -= 1
            new_k += 1
            if time.time() > deadline - 5:
                break
            
        # Extend Right
        while new_s + new_k <= 10**12:
            next_val = phi_fast(new_s + new_k)
            if next_val in current_vals:
                break
            current_vals.add(next_val)
            new_k += 1
            if time.time() > deadline - 5:
                break
                
        return new_s, new_k

    while time.time() < deadline:
        s = random.randint(1, 10**12)
        
        # Check forward run from s
        vals = []
        max_check = 50
        found_len = 0
        for i in range(max_check):
            if time.time() > deadline - 2:
                break
            v = phi_fast(s + i)
            if v in vals:
                found_len = i
                break
            vals.append(v)
            found_len = i + 1
        else:
            found_len = max_check
            
        if found_len > best_k:
            # Extend
            final_s, final_k = extend_run(s, found_len, set(vals[:found_len]))
            if final_k > best_k:
                best_k = final_k
                best_s = final_s
        
        # Also check if s is in the middle of a run?
        # We can check backward from s
        # But extend_run already checks left if we start from a valid prefix.
        # However, if the run started before s, we only found the suffix.
        # Example: Run is 1..10. We pick s=5.
        # We find phi(5)..phi(10) distinct (len 6).
        # extend_run(5, 6) will extend left to 1.
        # So this covers it.
        
        # But what if we pick s=11 (outside)?
        # Then we find len 0 or 1.
        # We might miss the run 1..10 if we don't pick s in [1,10].
        # Random sampling over 10^12 is sparse.
        # However, the baseline scan covers 1..200k.
        # For large ranges, the probability of hitting a specific run of length ~10 is low.
        # But we are looking for the MAX length.
        # If a run of length 20 exists, it's a needle in a haystack.
        # We rely on the fact that if such runs exist, they might be more frequent or we get lucky.
        # Or, we can scan larger ranges sparsely.
        
        # Alternative: Scan 10^6 to 2*10^6?
        # 10^6 numbers * phi cost.
        # If phi cost is 0.1ms, 100s. Too slow.
        # So random sampling is the only way for large N.

    return {'start': best_s, 'length': best_k}
