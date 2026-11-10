import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, p in enumerate(is_prime) if p]
    
    W = 200000  # Window size
    s = 10**10  # Start searching from 10^10
    
    best_k = 1
    best_s = 1
    
    while time.time() < deadline and s + W <= 10**12:
        # Segmented phi sieve for [s, s+W-1]
        phi = [0] * W
        rem = [0] * W
        for i in range(W):
            phi[i] = s + i
            rem[i] = s + i
        
        for p in primes:
            if p * p > s + W - 1:
                break
            # Find first multiple of p in [s, s+W-1]
            m = ((s + p - 1) // p) * p
            if m < s:
                m += p
            for j in range(m, s + W, p):
                idx = j - s
                phi[idx] = phi[idx] // p * (p - 1)
                while rem[idx] % p == 0:
                    rem[idx] //= p
        
        for i in range(W):
            if rem[i] > 1:
                phi[i] = phi[i] // rem[i] * (rem[i] - 1)
        
        # Find longest run of distinct phi values
        seen = set()
        cur_k = 0
        cur_s = s
        for i in range(W):
            if phi[i] in seen:
                cur_k = 0
                cur_s = s + i + 1
            else:
                seen.add(phi[i])
                cur_k += 1
                if cur_k > best_k:
                    best_k = cur_k
                    best_s = cur_s
                if best_k == 200:
                    return {'start': best_s, 'length': best_k}
            # Reset seen if run breaks? No, seen must be for the current run only.
            # Wait, the above logic is wrong. `seen` should be reset when the run breaks.
        
        # Correct scan for longest run
        seen = set()
        cur_k = 0
        cur_start = s
        for i in range(W):
            if phi[i] in seen:
                cur_k = 0
                cur_start = s + i + 1
                seen = set()
            else:
                seen.add(phi[i])
                cur_k += 1
                if cur_k > best_k:
                    best_k = cur_k
                    best_s = cur_start
                if best_k == 200:
                    return {'start': best_s, 'length': best_k}
        
        s += W
    
    return {'start': best_s, 'length': best_k}
