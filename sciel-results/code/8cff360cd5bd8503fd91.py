"""Rewritten program: fast divisor-sum sieve + targeted ratio search."""
import time
import math
import numpy as np
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    MAX_N = 10000000

    # --- Step 1: Compute sigma(n) for all n <= MAX_N using NumPy sieve ---
    sigma_arr = np.zeros(MAX_N + 1, dtype=np.int64)
    
    t0 = time.time()
    for i in range(1, MAX_N + 1):
        sigma_arr[i::i] += i
        # Periodic time check to avoid overshooting
        if i % 1000000 == 0:
            if time.time() > deadline - 10:
                # If sieve is taking too long, we need to be careful.
                # But we can't really stop the sieve halfway and have a valid sigma.
                # So we just let it finish if we're close.
                pass

    elapsed_sieve = time.time() - t0

    if time.time() > deadline - 5:
        # Not enough time to finish sieve and search
        # Return a known valid pair: sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = 0.0833...
        return [11, 6]

    # --- Step 2: Group numbers by sigma value ---
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        s = sigma_arr[n]
        fibers[s].append(n)
        
    if time.time() > deadline - 2:
        return [11, 6]

    # --- Step 3: Search for best pair ---
    # We want to minimize |a/b - 1.5|.
    # This is equivalent to minimizing |2a - 3b| / (2b).
    # For a fixed fiber, the numbers are fixed.
    # We should prioritize fibers that contain large numbers, as larger b reduces the error denominator.
    # However, we don't know which fibers have large numbers without checking.
    # We can just iterate through all fibers.
    
    # To speed up the search, we can sort the fibers by the maximum element in descending order?
    # That would require extra work.
    # Let's just iterate.
    
    # Also, we can skip fibers where the max element is small, because
    # err >= |2a - 3b| / (2 * max_b).
    # If 2 * max_b is small, the error is likely large.
    # But we don't know |2a - 3b| without checking.
    # However, if max_b is small, say < 1000, the error is at least 1/(2000) = 0.0005.
    # Our current best_err is ~0.0833.
    # So we only care about fibers with max_b > 1 / (2 * best_err) ~ 6.
    # This is a very weak filter.
    
    # Let's just do the bisect search.
    
    # To avoid checking the same pair twice (a,b) and (b,a),
    # we can just check all a for each b.
    # Since a and b are from the same set, and we check a closest to 1.5b,
    # we will find the best a for each b.
    
    # We can break early if best_err becomes very small?
    # The minimum possible non-zero error is 1/(2*b).
    # If we find an error of 1/(2*b_max_possible), we can't do better for that b.
    # But we might find a smaller error for a larger b in another fiber.
    
    # Let's iterate through fibers.
    # To make it faster, we can convert the fiber lists to arrays?
    # Bisect works on lists.
    
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5)
    
    for s_val, nums in fibers.items():
        if time.time() > deadline:
            break
            
        if len(nums) < 2:
            continue
            
        # Optional: skip if max(nums) is too small to improve current best
        # max_possible_improvement = 1 / (2 * max(nums))
        # if max_possible_improvement >= best_err: continue
        # This is not quite right because |2a-3b| could be 1.
        # err = 1 / (2b).
        # So if 1 / (2 * max(nums)) >= best_err, we can't improve.
        # i.e., if max(nums) <= 1 / (2 * best_err), skip.
        
        max_b = max(nums)
        if max_b <= 1 / (2 * best_err):
            continue
            
        nums.sort()
        
        # For each b in nums, find a closest to 1.5 * b
        # We can iterate b in descending order to find large b first?
        # If we find a good error with large b, we can prune more.
        # But sorting nums is O(N log N). Iterating is O(N).
        # Let's just iterate.
        
        # To speed up, we can check b in descending order.
        # This way, we update best_err with larger b (smaller potential error) first.
        # This might allow pruning of other fibers earlier.
        
        for b in reversed(nums):
            # Check if current b can possibly improve best_err
            # Min error for this b is 1 / (2b) (if 2a-3b = +/- 1)
            # If 1 / (2b) >= best_err, we can't improve with this b.
            # But we might find a smaller error if 2a-3b is smaller? No, min non-zero is 1.
            # Wait, 2a-3b is an integer.
            # If 2a-3b = 0, error is 0.
            # If 2a-3b = 1, error is 1/(2b).
            # So min error for this b is 0 (if a=1.5b exists) or 1/(2b).
            # So if 1/(2b) >= best_err, we can skip this b ONLY IF we know 2a-3b != 0.
            # But we don't know that.
            # However, if best_err is already very small, and b is small, we can skip.
            
            if b <= 1 / (2 * best_err):
                # Even with error 0, we can't improve if best_err is 0? No.
                # If best_err > 0, and min possible error for this b is 1/(2b) (assuming no exact match),
                # and 1/(2b) >= best_err, then we can't improve UNLESS there is an exact match (error 0).
                # But if there was an exact match, we would have found it?
                # No, we are checking now.
                # So we must check for exact match.
                # But if we check for exact match, we might find error 0.
                # So we cannot skip based on 1/(2b) >= best_err if exact match is possible.
                # However, if best_err is 0, we can stop everything.
                # Let's just check.
                pass
            
            target_a = 1.5 * b
            pos = bisect.bisect_left(nums, target_a)
            
            # Check pos-1 and pos
            candidates = []
            if pos > 0:
                candidates.append(nums[pos-1])
            if pos < len(nums):
                candidates.append(nums[pos])
                
            for a in candidates:
                if a == b:
                    # a=b implies a/b=1, err=0.5.
                    # Unlikely to be optimal, but valid.
                    # We can skip if we have a better one.
                    # But let's just compute.
                    pass
                    
                err = abs(a / b - 1.5)
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
                    if err == 0.0:
                        return [int(best_a), int(best_b)]
                        
    return [int(best_a), int(best_b)]
