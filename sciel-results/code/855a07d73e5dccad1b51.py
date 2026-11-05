import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 80.0  # safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Precompute extracted[m] for m in [1, MAX_VAL]
    # extracted[m] = product of p^k for p < 24 where p^k || m
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)

    for p in small_primes:
        for m in range(p, MAX_VAL + 1, p):
            val = m
            while val % p == 0:
                extracted[m] *= p
                val //= p

    # Map extracted values to small integer IDs
    unique_vals, inverse_indices = np.unique(extracted[1:], return_inverse=True)

    # id_array[m] = ID for extracted[m], 1-based indexing
    id_array = np.empty(MAX_VAL + 1, dtype=np.int32)
    id_array[0] = -1
    id_array[1:] = inverse_indices

    K = len(unique_vals)

    # Vectorized sliding window approach
    # For each n, we need the number of distinct IDs in id_array[n+1 : n+1+WINDOW]
    # 
    # Strategy: Use a prefix-sum based approach with a trick for distinct counting.
    # Since K can be large, we use an alternative: for each position, track the 
    # "last seen" approach isn't directly vectorizable.
    #
    # Alternative: Since WINDOW is small (24), we can compute the distinct count
    # for all windows using a clever vectorized approach.
    #
    # For each window starting at position s (1-indexed for n+1), the window covers
    # id_array[s : s+WINDOW]. We want the number of distinct values.
    #
    # Approach: For each unique value v, find all positions where id_array == v.
    # For each window, count how many distinct values appear.
    #
    # Better approach for small WINDOW: 
    # Create a 2D array where row i represents window starting at N_MIN+i,
    # and compute distinct count. But this is O(N*WINDOW) which is ~24M, feasible.
    #
    # Let's use a different vectorized approach:
    # For each position j in [N_MIN+1, N_MAX+WINDOW], determine for which windows
    # it is the "first occurrence" of its value within that window.
    #
    # Actually, let's just do it efficiently:
    # For each value v, let positions be pos[0], pos[1], ..., pos[m-1]
    # A window [s, s+W-1] contains v if there exists i such that s <= pos[i] <= s+W-1
    # The window "first sees" v at position s if pos[i] is in [s, s+W-1] and 
    # pos[i-1] < s (or i=0)
    #
    # This is complex. Let's use a simpler O(N * WINDOW / word_size) approach:
    # Since WINDOW=24, we can pack the 24 IDs into a structure and count distinct.
    #
    # Simplest efficient approach: Use the fact that WINDOW is small.
    # For each window, we have 24 IDs. We can sort them and count distinct.
    # But doing this for 1M windows in a loop is what we want to avoid.
    #
    # Alternative: Use NumPy's ability to work with arrays.
    # Create a 2D array of shape (N_MAX-N_MIN+1, WINDOW) where each row is a window.
    # Then for each row, count distinct values.
    #
    # Memory: 1M * 24 * 4 bytes = ~96 MB, feasible.
    #
    # Let's try this approach.

    num_windows = N_MAX - N_MIN + 1
    
    # Create the 2D array of window IDs
    # Row i corresponds to n = N_MIN + i
    # Window covers id_array[N_MIN+i+1 : N_MIN+i+1+WINDOW]
    # Using stride tricks to create a view
    base = id_array[N_MIN+1 : N_MAX+WINDOW+1]
    # base has length N_MAX - N_MIN + WINDOW
    # Window i uses base[i : i+WINDOW]
    
    # Use np.lib.stride_tricks for a view
    from numpy.lib.stride_tricks import as_strided
    window_view = as_strided(base, shape=(num_windows, WINDOW), strides=(base.strides[0], base.strides[0]))
    
    # Now count distinct values in each row
    # For small WINDOW (24), we can sort each row and count changes
    # But sorting 1M rows of 24 elements... let's try
    
    # Alternative: For each row, use a hash-based distinct count
    # Since K might be large, let's use a different approach.
    
    # For each row, the distinct count is the number of unique elements.
    # We can compute this by: for each element, check if it equals any earlier element.
    
    # Let's use a vectorized approach with broadcasting for small WINDOW:
    # distinct_count[i] = number of j in 0..WINDOW-1 such that window_view[i,j] 
    #                     does not appear in window_view[i,0:j]
    
    # This is O(WINDOW^2) per row, so O(N * WINDOW^2) total = 1M * 576 = 576M ops
    # That might be too slow.
    
    # Better: sort each row and count distinct
    # np.sort on the last axis
    sorted_windows = np.sort(window_view, axis=1)
    
    # Count distinct: for each row, count positions where sorted_windows[i,j] != sorted_windows[i,j-1]
    # Plus 1 for the first element
    distinct_count = np.ones(num_windows, dtype=np.int32)
    for j in range(1, WINDOW):
        distinct_count += (sorted_windows[:, j] != sorted_windows[:, j-1]).astype(np.int32)
    
    # Find the minimum
    best_idx = np.argmin(distinct_count)
    best_n = N_MIN + best_idx
    best_count = distinct_count[best_idx]
    
    return best_n
