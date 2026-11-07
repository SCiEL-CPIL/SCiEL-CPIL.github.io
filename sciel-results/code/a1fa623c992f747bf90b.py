import math
import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s buffer for the 120s limit
    
    # Strategy: Use a sieve to compute k*sigma(k) for k up to some limit,
    # then find collisions (multiple k giving the same M).
    # This is a structural change from the smooth-number enumeration approach.
    
    # We need k <= sqrt(M) where M = k*sigma(k).
    # For k, sigma(k) is roughly k * log(log(k)) for large k.
    # So M ~ k^2 * log(log(k)).
    # If k ~ 10^7, M ~ 10^14 * ~3 = 3*10^14. Safe.
    # If k ~ 10^8, M ~ 10^16 * ~4 = 4*10^16. Safe.
    # If k ~ 10^9, M ~ 10^18 * ~5 = 5*10^18. Too big.
    # So we should restrict k to be roughly <= 5*10^8 to be safe.
    
    # But we can't sieve up to 5*10^8 in reasonable time.
    # Let's try k up to 10^7 first, which is 10 million.
    # Sieve of size 10^7 should be fast.
    
    # Actually, let's think about this differently.
    # We want to maximize the number of distinct k with the same M = k*sigma(k).
    # This is like finding collisions in the function f(k) = k*sigma(k).
    
    # Approach 1: Sieve up to N, compute f(k) for all k, count collisions.
    # N = 10^7: sieve is fast, but f(k) values are large (up to ~10^14).
    # We can use a dictionary to count.
    
    # Approach 2: Generate smooth numbers and compute f(k) for those.
    # This is what the current program does, but it's limited.
    
    # Let's try Approach 1 with N = 5*10^6 or 10^7.
    
    N = 10_000_000  # 10 million
    
    # Sieve to compute sigma(k) for all k up to N.
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Now compute f(k) = k * sigma(k) for all k.
    # Group by f(k) value.
    groups = defaultdict(list)
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            groups[M].append(k)
    
    # Find the group with the most elements.
    best_M = None
    best_k = []
    
    for M_val, ks in groups.items():
        if len(ks) > len(best_k):
            best_k = ks
            best_M = M_val
    
    # Time check
    if time.time() > deadline - 5.0:
        if best_M is None:
            best_M = 333312
            best_k = [336, 372, 434]
        else:
            best_M = int(best_M)
            best_k = [int(k) for k in best_k]
        return {'M': best_M, 'k': best_k}
    
    # If we didn't find many collisions, try a different approach.
    # Let's also try generating numbers of the form 2^a * 3^b * 5^c * 7^d
    # and computing f(k) for those.
    
    if len(best_k) < 5:
        # Generate more smooth numbers with 4 primes.
        max_k_val = 500_000_000
        
        pow2 = [1] * 30
        sig2 = [1] * 30
        for i in range(1, 30):
            pow2[i] = pow2[i-1] * 2
            sig2[i] = sig2[i-1] * 2 + 1
        
        pow3 = [1] * 20
        sig3 = [1] * 20
        for i in range(1, 20):
            pow3[i] = pow3[i-1] * 3
            sig3[i] = (pow3[i] * 3 - 1) // 2
        
        pow5 = [1] * 15
        sig5 = [1] * 15
        for i in range(1, 15):
            pow5[i] = pow5[i-1] * 5
            sig5[i] = (pow5[i] * 5 - 1) // 4
        
        pow7 = [1] * 12
        sig7 = [1] * 12
        for i in range(1, 12):
            pow7[i] = pow7[i-1] * 7
            sig7[i] = (pow7[i] * 7 - 1) // 6
        
        collisions = defaultdict(list)
        
        for a in range(30):
            if pow2[a] > max_k_val:
                break
            for b in range(20):
                if pow2[a] * pow3[b] > max_k_val:
                    break
                for c in range(15):
                    if pow2[a] * pow3[b] * pow5[c] > max_k_val:
                        break
                    for d in range(12):
                        k_part = pow2[a] * pow3[b] * pow5[c] * pow7[d]
                        if k_part > max_k_val:
                            break
                        
                        sigma_k = sig2[a] * sig3[b] * sig5[c] * sig7[d]
                        M = k_part * sigma_k
                        
                        if M > 10**18:
                            break
                        
                        collisions[M].append(k_part)
                        
                        if time.time() > deadline - 5.0:
                            break
                    if time.time() > deadline - 5.0:
                        break
                if time.time() > deadline - 5.0:
                    break
            if time.time() > deadline - 5.0:
                break
        
        # Merge with previous results
        for M_val, ks in collisions.items():
            if M_val in groups:
                groups[M_val].extend(ks)
            else:
                groups[M_val] = ks
        
        # Find the best again
        best_M = None
        best_k = []
        for M_val, ks in groups.items():
            if len(ks) > len(best_k):
                best_k = ks
                best_M = M_val
    
    # Fallback
    if best_M is None or len(best_k) < 3:
        best_M = 333312
        best_k = [336, 372, 434]
    
    # Ensure types
    best_M = int(best_M)
    best_k = [int(k) for k in best_k]
    
    # Final sanity check
    if best_M > 10**18:
        best_M = 333312
        best_k = [336, 372, 434]
    
    sqrt_M = math.isqrt(best_M)
    valid_k = [k for k in best_k if k <= sqrt_M]
    
    if len(valid_k) < len(best_k):
        best_k = valid_k
    
    if len(best_k) == 0:
        best_M = 333312
        best_k = [336, 372, 434]
    
    return {'M': best_M, 'k': best_k}
