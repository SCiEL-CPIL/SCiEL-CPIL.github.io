import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    TIME_LIMIT = 118.0

    def time_left():
        return TIME_LIMIT - (time.time() - start_time)

    # Sieve primes up to 10^6
    limit = 10**6
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    def compute_phi_segment(start, length):
        if length <= 0:
            return np.array([], dtype=np.int64)
        
        phi_vals = np.arange(start, start + length, dtype=np.int64)
        rem = np.arange(start, start + length, dtype=np.int64)
        
        for p in primes:
            first = ((start + p - 1) // p) * p
            if first >= start + length:
                continue
                
            indices = np.arange(first, start + length, p, dtype=np.int64) - start
            if len(indices) == 0:
                continue
            
            # Vectorized update for phi
            phi_vals[indices] = phi_vals[indices] // p * (p - 1)
            
            # Update rem
            # We need to divide rem by p as many times as possible
            # This is hard to vectorize fully without loops, but 
            # we can do a simple loop for the indices if count is small,
            # or use a while loop with masks.
            
            # Since p is prime, and we are in a small range,
            # the exponent of p in rem[i] is usually 1 or 2.
            # Let's do a simple loop for correctness and simplicity,
            # as the number of indices is small (length/p).
            for idx in indices:
                while rem[idx] % p == 0:
                    rem[idx] //= p

        # Handle large prime factors
        mask = rem > 1
        if np.any(mask):
            phi_vals[mask] = phi_vals[mask] // rem[mask] * (rem[mask] - 1)
            
        return phi_vals

    def find_best_run(phis, start):
        best_k = 1
        best_s = start
        left = 0
        freq = {}
        
        for right in range(len(phis)):
            val = phis[right]
            freq[val] = freq.get(val, 0) + 1
            
            while freq[val] > 1:
                left_val = phis[left]
                freq[left_val] -= 1
                if freq[left_val] == 0:
                    del freq[left_val]
                left += 1
            
            current_k = right - left + 1
            if current_k > best_k:
                best_k = current_k
                best_s = start + left
            if best_k >= 200:
                break
                
        return best_k, best_s

    best_k = 1
    best_s = 1

    # Check small numbers first
    if time_left() > 5:
        phis = compute_phi_segment(1, 1000)
        k, s = find_best_run(phis, 1)
        if k > best_k:
            best_k, best_s = k, s

    random.seed(42)
    seg_len = 200
    
    while time_left() > 2:
        start = random.randint(10**6, 10**12 - seg_len)
        phis = compute_phi_segment(start, seg_len)
        k, s = find_best_run(phis, start)
        if k > best_k:
            best_k, best_s = k, s

    return {'start': best_s, 'length': best_k}
