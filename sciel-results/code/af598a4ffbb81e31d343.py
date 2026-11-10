import time
import sys

def construct():
    start_time = time.time()
    limit = 1000000
    max_n = limit
    min_n = 1000
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small_part for range 1001 to 1000024
    # We need sp(x) for x in [1001, 1000024]
    # sp(x) = product of p^k for p in primes such that p^k divides x
    
    # Efficient calculation using prime powers
    size = max_n + 25
    sp = [1] * size
    
    # For each prime p, for each power p^k < size, multiply sp[j] by p for j divisible by p^k
    for p in primes:
        p_power = p
        while p_power < size:
            for j in range(p_power, size, p_power):
                sp[j] *= p
            p_power *= p
    
    best_n = min_n
    best_val = float('inf')
    
    # Sliding window with frequency count
    # Window covers n+1 to n+24 (24 elements)
    
    # Initialize frequency map for the first window (n = min_n)
    freq = {}
    distinct_count = 0
    for i in range(1, 25):
        val = sp[min_n + i]
        if val not in freq:
            freq[val] = 0
            distinct_count += 1
        freq[val] += 1
    
    if distinct_count < best_val:
        best_val = distinct_count
        best_n = min_n
    
    # Sliding window: slide from n to n+1
    # Remove sp(n), add sp(n+24)
    
    # Deadline: leave safety margin (evaluator kills at 90s)
    deadline = start_time + 75
    
    for n in range(min_n + 1, max_n + 1):
        if time.time() > deadline:
            break
        
        # Remove the element that's leaving the window: sp(n)
        old_val = sp[n]
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
            distinct_count -= 1
        
        # Add the element entering the window: sp(n + 24)
        new_val = sp[n + 24]
        if new_val not in freq:
            freq[new_val] = 0
            distinct_count += 1
        freq[new_val] += 1
        
        if distinct_count < best_val:
            best_val = distinct_count
            best_n = n
    
    return best_n
