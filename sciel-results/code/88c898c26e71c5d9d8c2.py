import numpy as np
import time
import sys
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds margin
    
    N = 10**7
    
    # 1. Compute sigma(n) for n=1..N using Linear Sieve
    # This is O(N) and should take ~2-3 seconds in Python.
    
    sigma = [0] * (N + 1)
    sigma[1] = 1
    power = [0] * (N + 1)
    sigma_power = [0] * (N + 1)
    is_prime = [True] * (N + 1)
    primes = []
    
    # Local variables for speed
    sig = sigma
    pow_ = power
    spow = sigma_power
    isp = is_prime
    prm = primes
    N_local = N
    
    for i in range(2, N_local + 1):
        if isp[i]:
            prm.append(i)
            sig[i] = 1 + i
            pow_[i] = i
            spow[i] = 1 + i
        
        for p in prm:
            ip = i * p
            if ip > N_local:
                break
            isp[ip] = False
            if i % p == 0:
                pow_[ip] = pow_[i] * p
                spow[ip] = spow[i] + pow_[ip]
                sig[ip] = sig[i] // spow[i] * spow[ip]
                break
            else:
                pow_[ip] = p
                spow[ip] = 1 + p
                sig[ip] = sig[i] * (1 + p)
                
    if time.time() > deadline - 20:
        return [11, 6]

    # 2. Convert to NumPy array for fast argsort
    # sig is a list of Python ints. Converting to np.array is fast.
    sigma_np = np.array(sig, dtype=np.int64)
    
    # 3. Sort indices by sigma value
    # argsort returns indices i such that sigma_np[i] is sorted
    sorted_indices = np.argsort(sigma_np)
    
    if time.time() > deadline - 10:
        return [11, 6]

    # 4. Iterate through sorted indices to find groups
    best_a, best_b = 11, 6
    best_err = abs(11/6 - 1.5)
    
    # We will iterate through the sorted array.
    # A group is a contiguous sequence of indices with the same sigma value.
    
    # To speed up, we can process the array in chunks or just loop.
    # Looping 10M times in Python is ~0.5-1s.
    
    # We need to be careful with memory. sorted_indices is 10M int64s (80MB).
    # sigma_np is 10M int64s (80MB).
    
    # Optimization: Skip groups that are too small or have poor potential.
    
    i = 1
    n_sorted = len(sorted_indices)
    
    while i < n_sorted:
        # Check time
        if time.time() > deadline - 2:
            break
            
        # Find the end of the current group
        current_sigma = sigma_np[sorted_indices[i]]
        j = i + 1
        while j < n_sorted and sigma_np[sorted_indices[j]] == current_sigma:
            j += 1
            
        # Group is sorted_indices[i:j]
        group_size = j - i
        
        if group_size >= 2:
            # Extract the numbers in this group
            # These are the values of n, not the indices in the sorted array
            # Wait, sorted_indices[k] is the index n.
            # So the numbers are sorted_indices[i:j]
            
            nums = sorted_indices[i:j]
            
            # Optional: Prune based on range
            min_num = nums[0] # Not necessarily min value, just first in sorted-by-sigma order
            # We need min and max of the VALUES
            # Since nums are not sorted by value, we need to find min/max
            # Or we can just sort them.
            
            # Sort the numbers in the group
            # nums is a numpy array slice. We can convert to list and sort, or use np.sort
            # np.sort is faster for large arrays, but for small groups, list sort is fine.
            # Most groups are small.
            
            if group_size < 1000:
                nums_list = nums.tolist()
                nums_list.sort()
            else:
                nums_list = np.sort(nums).tolist()
                
            # Check range potential
            min_val = nums_list[0]
            max_val = nums_list[-1]
            
            # If the ratio max/min is such that 1.5 is far away, skip?
            # Best possible error is dist(1.5, [min/max, max/min])
            # If min/max > 1.5, then a/b < 1.5 for all a<=max, b>=min?
            # No, a/b can be anything in [min/max, max/min].
            # We want a/b close to 1.5.
            # If 1.5 < min/max, then a/b > 1.5 always. Best error is min/max - 1.5.
            # If 1.5 > max/min, then a/b < 1.5 always. Best error is 1.5 - max/min.
            # If min/max <= 1.5 <= max/min, best error is 0 (potentially).
            
            ratio_min = min_val / max_val
            ratio_max = max_val / min_val
            
            if ratio_max < 1.5:
                # All ratios < 1.5
                potential_err = 1.5 - ratio_max
            elif ratio_min > 1.5:
                # All ratios > 1.5
                potential_err = ratio_min - 1.5
            else:
                potential_err = 0.0
                
            if potential_err >= best_err:
                i = j
                continue
                
            # Search for best pair in this group
            # For each b in nums_list, find a closest to 1.5 b
            
            # Use bisect
            # nums_list is sorted
            
            for b in nums_list:
                target_a = 1.5 * b
                pos = bisect.bisect_left(nums_list, target_a)
                
                # Check pos
                if pos < group_size:
                    a = nums_list[pos]
                    if a != b:
                        err = abs(a/b - 1.5)
                        if err < best_err:
                            if err != 0:
                                best_err = err
                                best_a, best_b = a, b
                # Check pos - 1
                if pos - 1 >= 0:
                    a = nums_list[pos - 1]
                    if a != b:
                        err = abs(a/b - 1.5)
                        if err < best_err:
                            if err != 0:
                                best_err = err
                                best_a, best_b = a, b
                            
        i = j

    return [int(best_a), int(best_b)]
