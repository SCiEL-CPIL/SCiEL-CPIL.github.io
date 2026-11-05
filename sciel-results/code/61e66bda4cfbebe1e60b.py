import time
from math import gcd
from functools import reduce

def construct():
    import sys
    sys.setrecursionlimit(10000)
    
    # Denominators 2 to 48
    denoms = list(range(2, 49))
    N = len(denoms)
    
    # Calculate LCM of 2..48
    # LCM(a, b) = abs(a*b) // gcd(a, b)
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    M = reduce(lcm, denoms)
    
    # Values in integer domain
    values = [M // d for d in denoms]
    
    # Suffix sums for pruning
    # suffix[i] = sum(values[i:])
    suffix = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix[i] = suffix[i + 1] + values[i]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit, 10s buffer
    
    # Initial best solution using greedy (seed's logic)
    best_sum = 0
    best_mask = 0
    current_sum = 0
    for i in range(N):
        if current_sum + values[i] < M:
            current_sum += values[i]
            best_mask |= (1 << i)
    best_sum = current_sum
    
    # If we can't find anything better, return the greedy one
    # But we will try to improve with B&B
    
    # Branch and Bound
    # We use a stack to avoid recursion depth issues and overhead, 
    # or just recursion since N=47 is small.
    # Recursion is simpler to write correctly.
    
    def search(i, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        # Time check
        if time.time() > deadline:
            raise TimeoutError()
            
        if i == N:
            return
            
        # Pruning: if even taking all remaining doesn't beat best, stop
        # Note: suffix[i] is sum of values from i to N-1
        if current_sum + suffix[i] <= best_sum:
            return
            
        # Option 1: Include item i
        # Check if it fits strictly below 1 (M)
        if current_sum + values[i] < M:
            new_sum = current_sum + values[i]
            new_mask = current_mask | (1 << i)
            
            # Update best if this partial solution is better
            # Note: We can update best even if we don't go deeper, 
            # but usually we go deeper to add more items.
            # However, if we stop here, the sum is new_sum.
            # But we can always add more items later.
            # So we only update best if we find a complete valid set?
            # No, any valid set is a candidate. 
            # The set defined by current_mask is valid if sum < M.
            # So if new_sum > best_sum, update.
            if new_sum > best_sum:
                best_sum = new_sum
                best_mask = new_mask
                
            search(i + 1, new_sum, new_mask)
            
        # Option 2: Exclude item i
        search(i + 1, current_sum, current_mask)
        
    try:
        search(0, 0, 0)
    except TimeoutError:
        pass
    except RecursionError:
        pass

    # Convert best_mask to indices
    indices = [i for i in range(N) if (best_mask >> i) & 1]
    
    return indices
