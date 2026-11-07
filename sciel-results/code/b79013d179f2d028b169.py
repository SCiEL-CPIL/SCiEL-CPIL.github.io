import math
import time
import bisect
import sys

def construct():
    start_time = time.time()
    # Deadline: 90s limit. Leave 5s for cleanup/return.
    deadline = start_time + 85.0
    
    MAX = 10000000
    
    # 1. Sieve for smallest prime factor (SPF)
    # Using a list of ints. 
    # Optimization: Only sieve up to MAX.
    spf = list(range(MAX + 1))
    # Mark evens
    for i in range(4, MAX + 1, 2):
        spf[i] = 2
    # Sieve odd numbers
    limit = int(MAX**0.5) + 1
    for i in range(3, limit, 2):
        if spf[i] == i: # i is prime
            for j in range(i*i, MAX + 1, 2*i):
                if spf[j] == j:
                    spf[j] = i
                    
    if time.time() > deadline:
        return [11, 6]

    # 2. Compute sigma(n) for all n
    # sigma(n) is multiplicative.
    # If n = p^k * m, gcd(p,m)=1, sigma(n) = sigma(p^k) * sigma(m) = (p^(k+1)-1)/(p-1) * sigma(m)
    # We use DP:
    # Let spf[n] = p.
    # If n = p * m, and p does not divide m, sigma(n) = sigma(m) * (p+1)
    # If n = p * m, and p divides m, let m = p^k * r. Then n = p^(k+1) * r.
    # We store power[n] = p^k (highest power of spf[n] dividing n)
    # and sigma[n].
    
    sigma = [0] * (MAX + 1)
    power = [0] * (MAX + 1) # power[n] stores p^k where p=spf[n]
    
    sigma[1] = 1
    power[1] = 1
    
    for i in range(2, MAX + 1):
        p = spf[i]
        m = i // p
        if m % p == 0:
            # p divides m
            # power[m] is p^k
            # power[i] = power[m] * p
            pk = power[m] * p
            power[i] = pk
            # sigma[i] = sigma[m] + pk * sigma[m // power[m]]
            # Let r = m // power[m]
            r = m // power[m]
            sigma[i] = sigma[m] + pk * sigma[r]
        else:
            # p does not divide m
            power[i] = p
            sigma[i] = sigma[m] * (p + 1)
            
        # Check time periodically
        if i % 1000000 == 0 and time.time() > deadline:
            # Fallback to best known if we haven't finished
            return [11, 6]

    if time.time() > deadline:
        return [11, 6]

    # 3. Group numbers by sigma value
    # To save memory and time, we can just iterate and check?
    # No, grouping is necessary to find pairs with same sigma.
    # Dictionary mapping sigma -> list of numbers
    
    # Optimization: We only care about numbers that could form a good ratio.
    # But we don't know which sigma values will have many members.
    # Let's just group all.
    
    groups = {}
    for i in range(1, MAX + 1):
        s = sigma[i]
        if s in groups:
            groups[s].append(i)
        else:
            groups[s] = [i]
            
    if time.time() > deadline:
        return [11, 6]

    # 4. Search for best pair
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # This is equivalent to minimizing |2a - 3b| / b.
    # For a fixed group (same sigma), we have a set of numbers.
    # We want to find a, b in this set such that a/b is close to 1.5.
    
    best_a = 11
    best_b = 6
    best_num = 4 # |2*11 - 3*6| = 4
    best_den = 12 # 2*6
    
    # Sort groups by size? Or just iterate.
    # Iterating all groups might be slow if there are many.
    # Number of distinct sigma values is less than MAX.
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        if time.time() > deadline:
            break
            
        nums.sort()
        
        # For each b in nums, find a in nums closest to 1.5 * b
        # Since nums is sorted, we can use bisect.
        # To speed up, we can iterate b and binary search a.
        
        for b in nums:
            target_a = 1.5 * b
            
            # Find insertion point
            idx = bisect.bisect_left(nums, target_a)
            
            candidates = []
            if idx < len(nums):
                candidates.append(nums[idx])
            if idx > 0:
                candidates.append(nums[idx-1])
                
            for a in candidates:
                if a == b:
                    continue
                diff = abs(2 * a - 3 * b)
                den = 2 * b
                # Compare diff/den with best_num/best_den
                # diff * best_den < best_num * den
                if diff * best_den < best_num * den:
                    best_num = diff
                    best_den = den
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
