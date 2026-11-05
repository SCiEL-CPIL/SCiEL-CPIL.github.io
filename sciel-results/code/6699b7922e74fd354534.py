"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Compute small-prime part for each number up to max_val
    # small_part[m] = product of p^a for p in small_primes where p^a || m
    small_part = np.ones(max_val + 1, dtype=np.int64)
    small_part[0] = 0
    
    for p in small_primes:
        # For each multiple of p, compute p-adic valuation and multiply
        # Use vectorized approach: for each power p^k, mark multiples
        p_power = p
        while p_power <= max_val:
            # For multiples of p_power but not p_power * p, add one more factor
            # Actually, simpler: iterate and divide
            for m in range(p, max_val + 1, p):
                while m % p == 0:
                    m //= p
                    # This doesn't work in-place on numpy array
            break  # fallback to loop
        
        # Use a loop-based approach but optimized
        for m in range(p, max_val + 1, p):
            temp = m
            count = 0
            while temp % p == 0:
                count += 1
                temp //= p
            if count > 0:
                small_part[m] *= p ** count
    
    # Now small_part[m] is the complete factor of m composed of primes < 24
    # Range of interest: indices 1001 to 1000024
    # For n in [1000, 1000000], window is small_part[n+1 : n+25]
    
    # Extract the relevant slice
    # n=1000 -> indices 1001..10024
    # n=1000000 -> indices 1000001..1000024
    # So we need small_part[1001 : 1000025]
    slice_start = 1001
    slice_end = 1000025  # exclusive
    relevant = small_part[slice_start:slice_end]
    
    # Map distinct values to ranks
    unique_vals, inverse = np.unique(relevant, return_inverse=True)
    num_unique = len(unique_vals)
    
    # For each unique value, compute prefix sum of occurrences
    # prefix[j] = number of occurrences of this value in relevant[0:j]
    # Then count in window [i, i+24) is prefix[i+24] - prefix[i]
    
    # Use a 2D array: prefix[k, j] = count of unique value k in relevant[0:j]
    # This might be too large if num_unique is big, but let's check
    # num_unique should be at most ~1000-2000
    
    # Create boolean matrix: present[k, j] = 1 if relevant[j] == k
    # Then prefix sum along axis 1
    # present is num_unique x len(relevant)
    # len(relevant) = 1000024, num_unique ~ 1000-2000
    # That's ~1-2 million booleans, which is fine
    
    # Actually, let's use a different approach: for each position j, we know its value rank inverse[j]
    # We want for each window start i (0-indexed in relevant), count distinct values in relevant[i:i+24]
    # 
    # Vectorized approach: create a 2D array where row i has 1s at positions where values in window i appear
    # But that's 1M x 24 which is 24M entries, manageable
    # 
    # Better: for each value k, find positions where it occurs, then for each window, check if any occurrence is in the window
    # 
    # Most efficient: use the prefix sum approach per value
    # distinct_count[i] = sum over k of (1 if value k appears in relevant[i:i+24] else 0)
    # 
    # For value k, let positions_k be sorted list of indices where relevant[j]==k
    # Value k appears in window [i, i+24) iff there exists j in positions_k with i <= j < i+24
    # i.e., iff there exists j in positions_k with max(0, i-23) <= j < i+24... 
    # More precisely: iff min position in positions_k that is >= i is < i+24
    # 
    # For each k, we can compute for which i's the value appears
    # Then sum over k
    
    # Let's just do it with the prefix sum approach using a 2D array if num_unique is small enough
    if num_unique <= 5000:
        # Create indicator matrix
        # present[k, j] = 1 if relevant[j] == k else 0
        # Use scatter
        present = np.zeros((num_unique, len(relevant)), dtype=np.int8)
        present[inverse, np.arange(len(relevant))] = 1
        
        # Prefix sum along axis 1
        # prefix[k, j] = sum of present[k, 0:j]
        prefix = np.cumsum(present, axis=1)
        # To get count in [i, i+24): prefix[k, i+24] - prefix[k, i]
        # We need prefix[k, i+24] for i in [0, len-24]
        # Create shifted prefix
        # count_in_window[k, i] = prefix[k, i+24] - prefix[k, i]
        # This is a 2D array of size num_unique x (len-24)
        # len-24 = 1000000, num_unique ~ 1000-2000, so ~1-2M entries, fine
        
        # Extract the needed parts
        # For window start i (0-indexed in relevant), corresponding to n = 1000 + i
        # Window covers relevant[i:i+24], which is small_part[1001+i : 1001+i+24]
        # n = 1000 + i, so i = n - 1000
        # i ranges from 0 to 999999 (n from 1000 to 1000000)
        
        # count_in_window[k, i] = prefix[k, i+24] - prefix[k, i]
        # prefix has shape (num_unique, len(relevant)) = (num_unique, 1000024)
        # We need prefix[:, 24:1000024] - prefix[:, 0:1000000]
        # This gives shape (num_unique, 1000000)
        
        counts = prefix[:, 24:] - prefix[:, :len(relevant) - 24]
        # counts[k, i] = number of occurrences of value k in window i
        # distinct_count[i] = number of k's with counts[k, i] > 0
        distinct_counts = np.sum(counts > 0, axis=0)
        
        best_i = np.argmin(distinct_counts)
        best_n = 1000 + best_i
        return best_n
    else:
        # Fallback to sliding window with dictionary
        freq = {}
        distinct_count = 0
        for i in range(24):
            val = relevant[i]
            freq[val] = freq.get(val, 0) + 1
            if freq[val] == 1:
                distinct_count += 1
        
        best_n = 1000
        best_distinct = distinct_count
        
        for i in range(24, len(relevant)):
            old_val = relevant[i - 24]
            freq[old_val] -= 1
            if freq[old_val] == 0:
                del freq[old_val]
                distinct_count -= 1
            
            new_val = relevant[i]
            freq[new_val] = freq.get(new_val, 0) + 1
            if freq[new_val] == 1:
                distinct_count += 1
            
            if distinct_count < best_distinct:
                best_distinct = distinct_count
                best_n = 1000 + i - 23
        
        return best_n

# EVOLVE-BLOCK-END
