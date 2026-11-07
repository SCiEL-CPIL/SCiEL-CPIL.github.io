import math
from collections import defaultdict

def _sigma(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    count = 0
    while temp % 2 == 0:
        count += 1
        temp //= 2
    if count > 0:
        result *= (2**(count+1) - 1)
    p = 3
    while p * p <= temp:
        if temp % p == 0:
            count = 0
            power = 1
            while temp % p == 0:
                count += 1
                temp //= p
                power *= p
            result *= (power * (p + 1)) // (p - 1)
        p += 2
    if temp > 1:
        result *= (temp + 1)
    return result

def construct():
    N = 10000000
    # Generate pool
    pool = set()
    max_k_sq = int(math.isqrt(N))
    for k in range(1, max_k_sq + 1):
        sq = k * k
        if sq <= N:
            pool.add(sq)
        two_sq = 2 * sq
        if two_sq <= N:
            pool.add(two_sq)
    
    # Compute sigmas and group
    groups = defaultdict(list)
    for n in pool:
        s = _sigma(n)
        groups[s].append(n)
        
    best_err = float('inf')
    best_a = 1
    best_b = 1
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        # Sort nums to help with finding close ratios?
        # Or just check all pairs since len is small
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                a = nums[i]
                b = nums[j]
                # Check both a/b and b/a?
                # The task says minimize |a/b - 3/2|.
                # Usually a and b are ordered in the pair [a,b].
                # The error is |a/b - 1.5|.
                # We should check both orientations?
                # The return is [a, b]. The evaluator calculates |a/b - 1.5|.
                # So if I find a pair {x, y}, I should check x/y and y/x.
                # Actually, the loop picks a and b. 
                # If I swap a and b, I get a different ratio.
                # I should check both |a/b - 1.5| and |b/a - 1.5|.
                
                err1 = abs(a / b - 1.5)
                if err1 == 0: continue
                if err1 < best_err:
                    best_err = err1
                    best_a = a
                    best_b = b
                    
                err2 = abs(b / a - 1.5)
                if err2 == 0: continue
                if err2 < best_err:
                    best_err = err2
                    best_a = b
                    best_b = a
    
    return [best_a, best_b]
