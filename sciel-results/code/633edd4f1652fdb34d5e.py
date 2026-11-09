import math
import time
import bisect
import numpy as np

MAX_N = 10_000_000
TIME_LIMIT = 55.0  # Safety margin for 60s limit

def construct():
    t_start = time.time()
    deadline = t_start + TIME_LIMIT

    # 1. Sieve sigma
    # Using int64 to hold sums. Max sigma for 10^7 is well within int64 range.
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d

    if time.time() > deadline:
        return [1, 1] # Fallback, though unlikely to be reached this fast

    # 2. Group indices by sigma value efficiently using sorting
    # We want to sort indices 1..MAX_N based on sigma[index]
    # sigma[0] is 0, we ignore index 0.
    indices = np.arange(1, MAX_N + 1, dtype=np.int64)
    sigma_vals = sigma[1:]  # Shape (MAX_N,)
    
    # Sort indices by their sigma values
    # argsort gives the permutation that sorts sigma_vals
    sort_order = np.argsort(sigma_vals, kind='stable')
    sorted_indices = indices[sort_order]
    sorted_sigmas = sigma_vals[sort_order]
    
    # 3. Identify groups in the sorted array
    # A group starts where sigma value changes
    # diff of sorted_sigmas will be 0 within a group, non-zero at boundaries
    # We can use np.diff to find boundaries
    # Note: np.diff prepends a 0 or we handle the first element
    # Let's create a mask of where a new group starts
    # diff[i] = sorted_sigmas[i] - sorted_sigmas[i-1]
    # If diff[i] != 0, then sorted_indices[i] starts a new group.
    # The first element (index 0) always starts a group.
    
    # Efficiently find unique sigma values and their counts?
    # Actually, we just need to binary search within the subarray for each b.
    # To do that, we need to know the start and end indices in `sorted_indices` for each sigma value.
    
    # Let's construct arrays `group_starts` and `group_ends` mapping sigma value to range in sorted_indices.
    # Since sigma values can be large, we can't use a direct array of size MAX_SIGMA.
    # We can use a dictionary mapping sigma -> (start, end) in the sorted array.
    
    # However, building a dict of 10^7 entries might be slow.
    # Alternative: For each b, we know its sigma. We can find its position in the sorted array?
    # No, that's hard without an inverse map.
    
    # Better approach:
    # We iterate b from MAX_N down to 1.
    # For a given b, we have sigma_b.
    # We need to find all a such that sigma_a == sigma_b.
    # In the sorted array, all elements with the same sigma are contiguous.
    # If we had a map: sigma_value -> (start_idx, end_idx) in sorted_indices, we could bisect.
    
    # Let's build this map. It has at most MAX_N entries (usually much fewer, as sigma values repeat).
    # Number of unique sigma values for N=10^7 is around ~1.5M to 2M?
    # Building a dict of 2M entries is fast in Python (<< 1s).
    
    # Find boundaries
    # diff = np.diff(sorted_sigmas)
    # starts = np.where(diff != 0)[0] + 1
    # starts = np.concatenate(([0], starts))
    # ends = np.concatenate((starts[1:], [len(sorted_indices)]))
    
    # Then for each unique sigma (sorted_sigmas[starts]), record the range.
    
    # Optimization: We can avoid building the full dict if we just iterate.
    # But we need random access for b.
    
    # Let's build the dict.
    diff = np.diff(sorted_sigmas)
    # Add a non-zero value at the beginning to mark the first start
    # Actually, simpler:
    # The starts are indices where sorted_sigmas[i] != sorted_sigmas[i-1], plus index 0.
    
    # Vectorized way to get starts and ends:
    # Create a mask of starts
    # mask[0] = True
    # mask[i] = sorted_sigmas[i] != sorted_sigmas[i-1]
    
    # np.diff returns array of size N-1.
    # We can do:
    # is_start = np.empty(len(sorted_sigmas), dtype=bool)
    # is_start[0] = True
    # is_start[1:] = diff != 0
    
    is_start = np.empty(len(sorted_sigmas), dtype=bool)
    is_start[0] = True
    is_start[1:] = diff != 0
    
    starts = np.where(is_start)[0]
    ends = np.empty(len(starts), dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = len(sorted_indices)
    
    # Now we have arrays `starts` and `ends` corresponding to the unique sigma values in `sorted_sigmas[starts]`.
    # unique_sigmas = sorted_sigmas[starts]
    
    # We need a mapping from sigma value to the index in the `starts`/`ends` arrays.
    # unique_sigmas are sorted. We can use np.searchsorted on unique_sigmas to find the position.
    
    unique_sigmas = sorted_sigmas[starts]
    
    # Precompute the search structure
    # For a given b, we compute s = sigma[b].
    # pos = np.searchsorted(unique_sigmas, s)
    # If pos < len(unique_sigmas) and unique_sigmas[pos] == s:
    #     start_idx = starts[pos]
    #     end_idx = ends[pos]
    #     subarray = sorted_indices[start_idx:end_idx]
    #     # Bisect in subarray
    
    # This is still a Python loop for the search, but the lookups are O(log U) where U is number of unique sigmas.
    # Bisecting in the subarray is O(log G) where G is group size.
    
    # Let's optimize the inner loop.
    # We can convert `starts`, `ends`, `unique_sigmas`, `sorted_indices` to lists or keep as numpy arrays.
    # Numpy array access is fast, but converting to list might be faster for small lookups?
    # Actually, keeping them as numpy arrays and using .item() or direct indexing is fine.
    # However, `bisect` module works on lists. We can pass a list slice? No, that copies.
    # We can use `bisect` on the numpy array directly? No, `bisect` expects a sequence supporting `__getitem__` and `__len__`. Numpy arrays support this.
    
    bisect_left = bisect.bisect_left
    
    best_err = float('inf')
    best_pair = (1, 1)
    
    # To speed up the search, we can iterate b from MAX_N downwards.
    # We can break early if 1/(2*b) >= best_err.
    
    # Pre-extract sigma values for b lookup?
    # sigma is a numpy array. sigma[b] is fast.
    
    # Let's run the search
    # We need to be careful with time.
    
    for b in range(MAX_N, 0, -1):
        if time.time() > deadline:
            break
            
        s_b = sigma[b]
        
        # Find group for s_b
        pos = np.searchsorted(unique_sigmas, s_b)
        
        if pos >= len(unique_sigmas) or unique_sigmas[pos] != s_b:
            continue
            
        start_idx = starts[pos]
        end_idx = ends[pos]
        
        # If group size is 1, no pair (a != b required, and only one element)
        if end_idx - start_idx < 2:
            continue
            
        # The subarray is sorted_indices[start_idx:end_idx]
        # We want a in this subarray such that a ~ 1.5 * b
        # target = 1.5 * b
        
        # We can bisect on the full sorted_indices using the range?
        # bisect_left(sorted_indices, target, start_idx, end_idx)
        
        target = 1.5 * b
        pos_a = bisect_left(sorted_indices, target, start_idx, end_idx)
        
        # Check pos_a and pos_a - 1
        
        # Candidate 1: pos_a
        if pos_a < end_idx:
            a = sorted_indices[pos_a]
            if a != b:
                # err = |a/b - 1.5| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                if num > 0:
                    err = num / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
        # Candidate 2: pos_a - 1
        if pos_a > start_idx:
            a = sorted_indices[pos_a - 1]
            if a != b:
                num = abs(2 * a - 3 * b)
                if num > 0:
                    err = num / (2.0 * b)
                    if err < best_err:
                        best_err = err
                        best_pair = (a, b)
                        
        # Early termination check
        # If the minimum possible error for any smaller b is >= best_err, we can stop.
        # Min error for b' is approx 1/(2*b').
        # For b' = b-1, min err is 1/(2*(b-1)).
        # If 1/(2*b) >= best_err, then for all b' <= b, 1/(2*b') >= 1/(2*b) >= best_err.
        if best_err > 0 and (1.0 / (2.0 * b)) >= best_err:
            break

    a, b = best_pair
    return [int(a), int(b)]
