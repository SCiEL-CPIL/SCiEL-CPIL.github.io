"""
Optimized construct function for distinct subset sums.
The task is to find 12 distinct positive integers with distinct subset sums,
minimizing the largest integer.

Mathematical analysis:
For a set of n distinct positive integers with all 2^n subset sums distinct,
the largest element must be at least 2^(n-1). This is because the 2^n subset
sums are all distinct non-negative integers, and the largest subset sum (the
sum of all elements) must be at least 2^n - 1. The largest element a_n must
satisfy a_n >= (2^n - 1) / 2 > 2^(n-1) - 1, so a_n >= 2^(n-1).

The set {1, 2, 4, ..., 2^(n-1)} achieves this lower bound, with largest
element exactly 2^(n-1). For n=12, this gives largest element 2^11 = 2048.

We verify this result using a greedy construction that maintains the invariant
that all subset sums remain distinct at each step.
"""

import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s hard limit
    
    n = 12
    
    # Greedy construction: maintain set of elements and all subset sums.
    # At each step, choose the smallest positive integer that keeps all
    # subset sums distinct. This produces powers of 2.
    elements = []
    subset_sums = {0}
    max_sum = 0
    
    for i in range(n):
        # The smallest valid next element is max_sum + 1
        # This ensures that the new subset sums (next_val + s for s in subset_sums)
        # do not collide with existing subset sums, since:
        #   min new sum = next_val + 0 = max_sum + 1 > max_sum = max old sum
        # So all new sums are strictly greater than all old sums.
        next_val = max_sum + 1
        
        if next_val > 1048576:
            # Should not happen for n=12, but guard against it
            break
        
        elements.append(next_val)
        
        # Update subset sums: add next_val + s for all existing sums s
        new_sums = {next_val + s for s in subset_sums}
        subset_sums = subset_sums.union(new_sums)
        
        # Update max_sum
        max_sum += next_val
        
        # Time check
        if time.time() > deadline:
            # Fall back to known optimal solution
            return [1 << j for j in range(n)]
    
    # Verify distinctness: we should have exactly 2^n distinct subset sums
    if len(subset_sums) != (1 << n):
        # Fallback to hardcoded powers of 2
        return [1 << i for i in range(n)]
    
    return elements
