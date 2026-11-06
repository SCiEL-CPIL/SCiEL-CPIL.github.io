import time
import math

def construct():
    deadline = time.time() + 75  # Safety margin before 90s limit
    target = 123457
    
    # Compute largest prime factor for all numbers up to target-1
    max_n = target - 1
    lpf = [0] * (max_n + 1)
    lpf[1] = 1
    for i in range(2, max_n + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, max_n + 1, i):
                lpf[j] = i
                
    # Primes list up to a reasonable limit
    # We can generate primes up to max_n if needed, but 97 is likely enough
    # If not, we can extend
    primes_list = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251]
    
    # Try increasing k until we find a solution
    for k in range(1, len(primes_list) + 1):
        if time.time() > deadline - 5:
            break
        
        p_k = primes_list[k-1]
        
        # Check if target can be written as sum of two numbers with max prime factor <= p_k
        # We scan a from 1 to target//2
        for a in range(1, target // 2 + 1):
            if lpf[a] <= p_k and lpf[target - a] <= p_k:
                return [a, target - a]
        
        # If we're getting close to deadline, do a fallback scan
        if time.time() > deadline - 10:
            break
            
    # Fallback: full scan to find best pair
    # This ensures we always return a valid answer
    best_max = float('inf')
    best_a = 1
    
    for a in range(1, target):
        if time.time() > deadline - 1:
            break
        b = target - a
        cur_max = max(lpf[a], lpf[b])
        if cur_max < best_max:
            best_max = cur_max
            best_a = a
            
    return [best_a, target - best_a]
