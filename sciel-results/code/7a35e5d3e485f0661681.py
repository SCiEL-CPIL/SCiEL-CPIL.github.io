import numpy as np
import time
import sys
from collections import defaultdict
import bisect

def construct():
    # Deadline guard: 85 seconds max (evaluator kills at 90s, leave margin)
    start_time = time.time()
    deadline = start_time + 85
    
    N = 10000000
    
    # 1. Compute sigma(n) for all n up to N using NumPy sieve
    # sigma[n] = sum of divisors of n
    # Initialize with 1 (divisor 1)
    sigma = np.ones(N + 1, dtype=np.int64)
    
    # Sieve: for each i, add i to all multiples of i
    # This is O(N log N) but vectorized operations are fast
    for i in range(1, N + 1):
        if time.time() > deadline - 10:
            # Fallback if sieve takes too long
            return [11, 6]
        sigma[i::i] += i
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 2. Group numbers by sigma value
    # Convert to list for faster access in grouping
    sigma_list = sigma.tolist()
    
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        s = sigma_list[n]
        sigma_groups[s].append(n)
        if n % 1000000 == 0 and time.time() > deadline - 5:
            break

    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for best pair (a, b) such that sigma(a) == sigma(b) and a/b ~ 1.5
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # Iterate over groups with at least 2 elements
    # Optimization: Sort groups by max value descending?
    # Large b gives smaller potential error.
    # But we don't know which groups have large b without checking.
    # Let's just iterate all groups.
    
    # To prioritize finding good solutions early for pruning,
    # we could sort groups by their maximum element in descending order.
    # This might help find a low best_error earlier, enabling better pruning.
    
    # However, sorting all groups might take time.
    # Let's stick to iterating and relying on the inner loop pruning.
    
    for s, nums in sigma_groups.items():
        if time.time() > deadline - 2:
            break
            
        if len(nums) < 2:
            continue
        
        # Sort nums for binary search
        nums_sorted = sorted(nums)
        
        # Iterate b in descending order to find large b (small error) first
        # This allows the pruning condition 1/(2b) >= best_error to break the loop
        # as soon as b becomes too small to improve the current best.
        
        for b in reversed(nums_sorted):
            # Pruning: if even the best possible a (error 1/(2b)) is not better than current best, skip
            if 1.0 / (2.0 * b) >= best_error:
                # Since b is decreasing, 1/(2b) is increasing.
                # So for all smaller b', 1/(2b') > 1/(2b) >= best_error.
                # We can break out of the b loop for this group.
                break
            
            target_a_float = 1.5 * b
            # Binary search for target_a in nums_sorted
            idx = bisect.bisect_left(nums_sorted, int(target_a_float))
            
            # Check candidates around idx
            # We want a such that |2a - 3b| is small.
            # a ≈ 1.5 b.
            # Let's check a window of indices around idx.
            # The error is |2a - 3b| / (2b).
            # For a fixed b, we want to minimize |2a - 3b|.
            # Since a is from nums_sorted, we check neighbors of the ideal a.
            
            # Window size:
            # If the gap between consecutive elements in nums_sorted is large,
            # a small window might miss the best a.
            # However, if the gap is large, the error |2a - 3b| will be large,
            # so it's unlikely to improve best_error if best_error is small.
            # Let's use a window of +/- 10 indices.
            # If the group is dense, this is fine.
            # If the group is sparse, we might miss a better a, but that a would have
            # a larger error anyway, so it might not be the global minimum.
            # Actually, if we miss the closest a, we might miss a very good solution
            # if the group is sparse but has a very large b.
            # But if the group is sparse, the chance of having a very large b AND
            # a very close a is low.
            # Let's expand the window slightly to be safe, e.g., +/- 20.
            
            start_idx = max(0, idx - 20)
            end_idx = min(len(nums_sorted), idx + 21)
            
            for i in range(start_idx, end_idx):
                a = nums_sorted[i]
                if a == b:
                    continue
                
                # Calculate error exactly
                err_num = abs(2 * a - 3 * b)
                if err_num == 0:
                    # Raw value 0 is invalid
                    continue
                
                err_den = 2 * b
                error = err_num / err_den
                
                if error < best_error:
                    best_error = error
                    best_a = a
                    best_b = b

    # Ensure types are Python ints
    return [int(best_a), int(best_b)]

# Test / Sanity Check if run directly
if __name__ == "__main__":
    res = construct()
    print(res)
    a, b = res
    if isinstance(a, int) and isinstance(b, int):
        print(f"a={a}, b={b}")
    else:
        print("Type error")
