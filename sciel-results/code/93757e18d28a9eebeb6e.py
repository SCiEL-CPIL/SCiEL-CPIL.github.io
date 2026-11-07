"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from bisect import bisect_left, bisect_right
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    N = 10000000
    
    # Compute sigma(n) for all n up to N using linear sieve
    sigma = [0] * (N + 1)
    primes = []
    is_composite = bytearray(N + 1)
    sigma[1] = 1
    
    for i in range(2, N + 1):
        if not is_composite[i]:
            primes.append(i)
            sigma[i] = 1 + i
        
        for p in primes:
            if i * p > N:
                break
            is_composite[i * p] = 1
            if i % p == 0:
                # i = p^k * m where gcd(m, p) = 1
                # Find the highest power of p dividing i
                temp = i
                p_power = 1
                while temp % p == 0:
                    temp //= p
                    p_power *= p
                # sigma(i*p) = sigma(i) * (1 + p + ... + p^(k+1)) / (1 + p + ... + p^k)
                # = sigma(i) * (p_power * (p+1) - p) / (p_power - 1) ... 
                # Actually: sigma(i) = sigma(m) * (p^(k+1)-1)/(p-1)
                # sigma(i*p) = sigma(m) * (p^(k+2)-1)/(p-1)
                # ratio = (p^(k+2)-1)/(p^(k+1)-1)
                # Let pk = p^k, then sigma(m) = sigma(i) * (p-1)/(pk+1-p)
                # sigma(i*p) = sigma(m) * (p*pk+1-p)/(p-1) = sigma(i) * (p*pk+1-p)/(pk+1-p) * ... 
                # Simpler: sigma(i*p) = sigma(i) - sigma(i/p^k) + sigma(i/p^k) * (1+p+...+p^(k+1))
                # Let m = i / p^k (p-free part)
                # sigma(i) = sigma(m) * (p^(k+1)-1)/(p-1)
                # sigma(i*p) = sigma(m) * (p^(k+2)-1)/(p-1)
                # So sigma(i*p) = sigma(i) * (p^(k+2)-1) / (p^(k+1)-1)
                # Compute without overflow:
                # (p^(k+2)-1) / (p^(k+1)-1) = 1 + p^(k+1)/(p^(k+1)-1)
                # Let's compute directly:
                pk = p_power  # p^k
                # sigma(i*p) = sigma(i) * (pk*p + 1 - p) / (pk + 1 - p)
                # Wait, let me re-derive:
                # i = m * p^k, p doesn't divide m
                # sigma(i) = sigma(m) * (p^(k+1) - 1) / (p - 1)
                # i*p = m * p^(k+1)
                # sigma(i*p) = sigma(m) * (p^(k+2) - 1) / (p - 1)
                # sigma(i*p) / sigma(i) = (p^(k+2) - 1) / (p^(k+1) - 1)
                # = (p * p^(k+1) - 1) / (p^(k+1) - 1)
                # = (p * (p^(k+1) - 1) + (p - 1)) / (p^(k+1) - 1)
                # = p + (p - 1) / (p^(k+1) - 1)
                # So sigma(i*p) = sigma(i) * p + sigma(i) * (p-1) / (p^(k+1)-1)
                # But this requires exact division. Let's use:
                # sigma(i*p) = sigma(i) * (p^(k+2) - 1) // (p^(k+1) - 1)
                # Since both are integers and the division is exact.
                # p^(k+1) - 1 = pk * p - 1
                # p^(k+2) - 1 = pk * p * p - 1
                # sigma(i*p) = sigma(i) * (pk * p * p - 1) // (pk * p - 1)
                pkp = pk * p  # p^(k+1)
                sigma[i * p] = sigma[i] * (pkp * p - 1) // (pkp - 1)
                break
            else:
                sigma[i * p] = sigma[i] * (1 + p)
    
    if time.time() > deadline - 10:
        return [11, 6]
    
    # Group numbers by sigma value
    # For each sigma value, we want to find pairs (a, b) minimizing |a/b - 3/2|
    # This is equivalent to minimizing |2a - 3b| / (2b)
    # For a fixed b, we want a close to 3b/2
    
    # Build groups: sigma_val -> sorted list of numbers
    groups = defaultdict(list)
    for n in range(1, N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    best_error_num = 10**18  # numerator of error (we'll compare fractions)
    best_error_den = 1
    best_pair = (11, 6)
    
    # Sort groups by size descending - larger groups have more chances
    group_items = [(s, nums) for s, nums in groups.items() if len(nums) >= 2]
    group_items.sort(key=lambda x: len(x[1]), reverse=True)
    
    # For each group, search for best pair
    for s_val, nums in group_items:
        if time.time() > deadline - 2:
            break
        
        n = len(nums)
        if n < 2:
            continue
        
        # For each b in the group, find a in the group closest to 3b/2
        # Since we want |a/b - 3/2| = |2a - 3b| / (2b), for fixed b we want a ≈ 3b/2
        # To minimize the overall error, we should prioritize larger b (larger denominator)
        # and look for a close to 3b/2
        
        # Process b values from largest to smallest (larger b gives smaller potential error)
        for idx in range(n - 1, -1, -1):
            if time.time() > deadline - 1:
                break
            
            b = nums[idx]
            target = 3 * b // 2
            
            # Binary search for target in nums
            pos = bisect_left(nums, target)
            
            # Check candidates around pos
            for p in range(max(0, pos - 5), min(n, pos + 6)):
                a = nums[p]
                if a == b:
                    continue
                # error = |2a - 3b| / (2b)
                err_num = abs(2 * a - 3 * b)
                err_den = 2 * b
                # Compare err_num/err_den with best_error_num/best_error_den
                # err_num * best_error_den < best_error_num * err_den
                if err_num * best_error_den < best_error_num * err_den:
                    best_error_num = err_num
                    best_error_den = err_den
                    best_pair = (a, b)
    
    a, b = best_pair
    return [int(a), int(b)]

# EVOLVE-BLOCK-END
