import time
import sys
import random
from math import gcd

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin
    
    N = 256
    vertices = list(range(2, N + 1))
    n = len(vertices)
    
    # Split vertices into small (2-16) and large (17-256)
    # Small numbers have high weight and high conflict probability.
    # Large numbers have low weight and low conflict probability.
    split_idx = 15  # vertices[15] is 16. indices 0-15 are 2-16.
    small_vertices = vertices[:split_idx+1]
    large_vertices = vertices[split_idx+1:]
    
    n_small = len(small_vertices)
    n_large = len(large_vertices)
    
    # Precompute conflicts within small
    # conflict_small[i] bitmask for small indices
    conflict_small = [0] * n_small
    for i in range(n_small):
        a = small_vertices[i]
        for j in range(i + 1, n_small):
            b = small_vertices[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict_small[i] |= (1 << j)
                conflict_small[j] |= (1 << i)
                
    # Precompute conflicts between small and large
    # conflict_small_large[i] bitmask for large indices that conflict with small[i]
    conflict_small_large = [0] * n_small
    for i in range(n_small):
        a = small_vertices[i]
        mask = 0
        for j in range(n_large):
            b = large_vertices[j]
            g = gcd(a, b)
            if a * b // g <= N:
                mask |= (1 << j)
        conflict_small_large[i] = mask
        
    # Precompute conflicts within large
    # We don't need a full bitmask matrix for large if we use a greedy list approach,
    # but for efficiency in checking validity of adding a large node, 
    # we can precompute a list of conflicts for each large node?
    # Actually, n_large is ~240. A full matrix is 240*240/2 ~ 28k pairs.
    # Let's precompute conflict_large[j] bitmask for large indices.
    conflict_large = [0] * n_large
    for i in range(n_large):
        a = large_vertices[i]
        for j in range(i + 1, n_large):
            b = large_vertices[j]
            g = gcd(a, b)
            if a * b // g <= N:
                conflict_large[i] |= (1 << j)
                conflict_large[j] |= (1 << i)

    weights_small = [1.0 / v for v in small_vertices]
    weights_large = [1.0 / v for v in large_vertices]
    
    best_val = 0.0
    best_small_mask = 0
    best_large_mask = 0
    
    # Enumerate all valid subsets of small vertices
    # There are 2^15 = 32768 possible masks. We can check validity quickly.
    # Valid if no two selected small vertices conflict.
    
    # Optimization: Precompute valid small masks? Or just iterate and check.
    # Checking validity of a mask m:
    # for each i in m, if (conflict_small[i] & m) != 0, invalid.
    # Since n_small is small, we can just iterate all 2^15 masks.
    
    # To speed up, we can generate valid masks recursively or just filter.
    # Let's filter.
    
    # Sort large vertices by weight descending to greedy pick best available
    # indices in large_vertices order
    large_order = sorted(range(n_large), key=lambda i: -weights_large[i])
    
    for sm in range(1 << n_small):
        # Check if sm is a valid independent set in small graph
        valid = True
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            if conflict_small[i] & sm:
                valid = False
                break
            m ^= lsb
            
        if not valid:
            continue
            
        # Calculate value of small part
        val_s = 0.0
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            val_s += weights_small[i]
            m ^= lsb
            
        # Determine which large vertices are forbidden by the selected small vertices
        forbidden_large = 0
        m = sm
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            forbidden_large |= conflict_small_large[i]
            m ^= lsb
            
        # Greedily select large vertices
        # We want to maximize sum of 1/l for l in large_vertices
        # such that l is not forbidden and l does not conflict with other selected large vertices.
        
        # Since large vertices have low weights, the optimal set is likely to be close to greedy.
        # We can do a local search on the large part for the top candidates?
        # Or just greedy?
        # Given time constraints, greedy is fast.
        # Let's try greedy first.
        
        selected_large = 0
        val_l = 0.0
        
        for j in large_order:
            if selected_large & (1 << j):
                continue
            if forbidden_large & (1 << j):
                continue
            # Check conflict with already selected large
            if conflict_large[j] & selected_large:
                continue
            
            # It is safe to add
            selected_large |= (1 << j)
            val_l += weights_large[j]
            
        total_val = val_s + val_l
        
        if total_val > best_val:
            best_val = total_val
            best_small_mask = sm
            best_large_mask = selected_large
            
    # Construct result
    result = []
    
    m = best_small_mask
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        result.append(small_vertices[i])
        m ^= lsb
        
    m = best_large_mask
    while m:
        lsb = m & -m
        j = lsb.bit_length() - 1
        result.append(large_vertices[j])
        m ^= lsb
        
    if not result:
        return [2]
        
    return result
