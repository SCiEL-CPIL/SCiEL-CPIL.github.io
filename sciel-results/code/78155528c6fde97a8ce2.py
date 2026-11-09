import time
import bisect
from math import lcm
import numpy as np

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = np.array(weights[:left_size], dtype=np.int64)
    right_weights = np.array(weights[left_size:], dtype=np.int64)
    
    # Generate all subset sums using iterative approach
    left_sums = np.zeros(1 << left_size, dtype=np.int64)
    for i in range(left_size):
        bit = 1 << i
        w = left_weights[i]
        left_sums[bit:bit*2:2] = left_sums[bit-1:bit*2-1:2] + w  # This won't work correctly
        
    # Actually, let's use a different approach with numpy
    # Generate sums using broadcasting
    left_sums = np.zeros(1 << left_size, dtype=np.int64)
    for i in range(left_size):
        half_size = 1 << i
        w = left_weights[i]
        left_sums[half_size:2*half_size] = left_sums[:half_size] + w
        
    right_sums = np.zeros(1 << right_size, dtype=np.int64)
    for i in range(right_size):
        half_size = 1 << i
        w = right_weights[i]
        right_sums[half_size:2*half_size] = right_sums[:half_size] + w
    
    # Create sorted indices for right sums
    right_order = np.argsort(right_sums)
    right_sorted_sums = right_sums[right_order]
    
    best_deficit = L
    best_left_mask = 0
    best_right_mask = 0
    
    start_time = time.time()
    deadline = start_time + 50.0
    
    # Convert to list for bisect or use numpy searchsorted
    for mask in range(1 << left_size):
        if mask % 100000 == 0 and time.time() > deadline:
            break
            
        ls = int(left_sums[mask])
        if ls >= L:
            continue
            
        limit = L - ls
        
        # Find right sum < limit
        idx = np.searchsorted(right_sorted_sums, limit, side='left') - 1
        
        if idx >= 0:
            rs = int(right_sorted_sums[idx])
            total = ls + rs
            deficit = L - total
            if deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = mask
                best_right_mask = int(right_order[idx])
                
    result_indices = []
    for i in range(left_size):
        if best_left_mask & (1 << i):
            result_indices.append(i)
    for i in range(right_size):
        if best_right_mask & (1 << i):
            result_indices.append(left_size + i)
            
    if not result_indices:
        result_indices = [0]
        
    return result_indices
