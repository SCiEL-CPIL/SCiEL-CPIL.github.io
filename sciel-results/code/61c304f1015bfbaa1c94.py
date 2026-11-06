import time
import random
import math

def construct():
    # Generate primes up to 10^6
    limit = 10**6
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, v in enumerate(is_prime) if v]
    
    deadline = time.time() + 110
    best_k = 1
    best_s = 1
    
    W = 200000  # Window size for phi computation
    
    def compute_phi_segment(L):
        """
        Computes phi for range [L, L+W-1]
        Returns list of phi values
        """
        R = L + W - 1
        rem = list(range(L, R+1))
        phi = list(range(L, R+1))
        
        for p in primes:
            if p * p > R:
                break
            # Find first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            # Iterate through multiples
            for j in range(start, R+1, p):
                idx = j - L
                if rem[idx] % p == 0:
                    # Divide out all p
                    while rem[idx] % p == 0:
                        rem[idx] //= p
                    phi[idx] -= phi[idx] // p
        
        # Handle remaining prime factors
        for i in range(W):
            if rem[i] > 1:
                phi[i] -= phi[i] // rem[i]
                
        return phi

    def find_best_in_phi(phi_vals, base_L):
        """
        Finds the longest run of distinct values in phi_vals
        Returns (k, start_offset)
        """
        seen = set()
        start = 0
        max_len = 0
        max_start = 0
        
        for i in range(len(phi_vals)):
            if phi_vals[i] in seen:
                # Collision at i
                # We need to move start forward until we remove the previous occurrence of phi_vals[i]
                while phi_vals[start] != phi_vals[i]:
                    seen.remove(phi_vals[start])
                    start += 1
                # Now phi_vals[start] == phi_vals[i]
                # We skip this value by moving start past it
                seen.remove(phi_vals[start])
                start += 1
            else:
                seen.add(phi_vals[i])
            
            current_len = i - start + 1
            if current_len > max_len:
                max_len = current_len
                max_start = start
                
        return max_len, base_L + max_start

    # Search in large ranges
    # We will try several random ranges
    num_tries = 10
    for _ in range(num_tries):
        if time.time() > deadline - 5:
            break
        
        # Random start in [10^11, 10^12 - W]
        L = random.randint(10**11, 10**12 - W)
        
        # Compute phi for this segment
        phi_vals = compute_phi_segment(L)
        
        # Find best run in this segment
        k, s = find_best_in_phi(phi_vals, L)
        
        if k > best_k:
            best_k = k
            best_s = s
            
            # If we found a very long run, we can stop or continue to find longer
            if best_k >= 200:
                break

    return {'start': best_s, 'length': best_k}
