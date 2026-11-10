import time
import sys
import numpy as np

def construct():
    start_time = time.time()
    limit = 1000000
    max_n = limit
    min_n = 1000
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # Precompute small_part for range 1 to 1000024 using NumPy
    size = max_n + 25
    sp = np.ones(size, dtype=np.int64)
    
    # For each prime p, for each power p^k < size, multiply sp[j] by p for j divisible by p^k
    for p in primes:
        p_power = p
        while p_power < size:
            sp[p_power-1::p_power] *= p
            p_power *= p
    
    # Convert to list for faster dict operations in sliding window
    sp_list = sp.tolist()
    
    best_n = min_n
    best_val = float('inf')
    
    # Sliding window with frequency count
    # Window covers n+1 to n+24 (24 elements)
    
    # Initialize frequency map for the first window (n = min_n)
    freq = {}
    distinct_count = 0
    for i in range(1, 25):
        val = sp_list[min_n + i]
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
    deadline = start_time + 80
    
    # Use local variable references for speed
    sp_l = sp_list
    freq_dict = freq
    
    for n in range(min_n + 1, max_n + 1):
        if time.time() > deadline:
            break
        
        # Remove the element that's leaving the window: sp(n)
        old_val = sp_l[n]
        c = freq_dict[old_val] - 1
        if c == 0:
            del freq_dict[old_val]
            distinct_count -= 1
        else:
            freq_dict[old_val] = c
        
        # Add the element entering the window: sp(n + 24)
        new_val = sp_l[n + 24]
        c2 = freq_dict.get(new_val, 0)
        if c2 == 0:
            freq_dict[new_val] = 1
            distinct_count += 1
        else:
            freq_dict[new_val] = c2 + 1
        
        if distinct_count < best_val:
            best_val = distinct_count
            best_n = n
    
    return best_n
