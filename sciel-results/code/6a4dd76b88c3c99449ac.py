import time
import random

def _is_valid(n, adj):
    # adj is a list of n ints, adj[i] is bitmask of neighbors of i
    # Check all subsets of size >= 6
    # Optimization: iterate masks, check bit_count >= 6
    # Use bit_count for degree calculation
    
    # Precompute bit counts for 16-bit chunks? No, int.bit_count is C speed.
    
    total_masks = 1 << n
    # We can iterate from 0 to total_masks
    # But we can skip masks with < 6 bits.
    # Generating combinations is slower than iterating all masks and filtering?
    # For n=21, 2^21 is small. Iterating all is fine.
    
    for m in range(total_masks):
        if m.bit_count() < 6:
            continue
        
        min_d = 6
        max_d = -1
        mm = m
        while mm:
            lsb = mm & -mm
            i = lsb.bit_length() - 1
            d = (adj[i] & m).bit_count()
            if d < min_d:
                min_d = d
            if d > max_d:
                max_d = d
            if min_d != max_d:
                break
            mm ^= lsb
        else:
            # If we finished the while loop without break, min_d == max_d
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try n=21
    for attempt in range(1000):
        if time.time() > deadline - 10.0:
            break
        n = 21
        adj = [0] * n
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        if _is_valid(n, adj):
            return {'n': n, 'edges': edges}
            
    # Try n=20
    for attempt in range(1000):
        if time.time() > deadline - 1.0:
            break
        n = 20
        adj = [0] * n
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        if _is_valid(n, adj):
            return {'n': n, 'edges': edges}
            
    # Fallback: return a known valid small graph
    # Empty graph on 5 vertices is always valid
    return {'n': 5, 'edges': []}
