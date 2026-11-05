import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0

    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    MAX_VAL = N_MAX + WINDOW

    # Compute extracted[m]
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        p_power = p
        while p_power <= MAX_VAL:
            extracted[p_power::p_power] *= p
            p_power *= p

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # Create windows
    # n ranges from N_MIN to N_MAX
    # Window for n is extracted[n+1 : n+1+WINDOW]
    # Indices in extracted: N_MIN+1 to N_MAX+WINDOW
    # We can use stride tricks to create a view of shape (N_MAX-N_MIN+1, WINDOW)
    from numpy.lib.stride_tricks import as_strided
    
    num_windows = N_MAX - N_MIN + 1
    # The data starts at extracted[N_MIN+1]
    start_idx = N_MIN + 1
    # Stride: 1 element in extracted is 8 bytes
    # We want a view where the first dimension steps by 1 in the original array
    # and the second dimension steps by 1 in the original array
    # Shape: (num_windows, WINDOW)
    # Strides: (8, 8)
    # But the start address is extracted[start_idx]
    
    # Note: as_strided requires careful handling of start address
    # It's safer to just use slicing if memory allows, but 10^6 * 24 is large.
    # Let's try the sliding window frequency count with NumPy optimization if possible, 
    # or just the Python loop which is known to be ~1s.
    
    # Given the time budget is 85s, the Python loop is safe.
    # The "structural change" requested is bit-packing or vectorization.
    # Let's try to vectorize the distinct count calculation using sorting.
    # For each window, sort the 24 elements and count changes.
    # This can be done with np.diff on sorted windows.
    
    # Create the 2D array of windows
    # This will allocate ~200MB.
    windows = np.lib.stride_tricks.sliding_window_view(extracted[start_idx:start_idx + num_windows + WINDOW - 1], WINDOW)
    
    # Sort each window
    # np.sort is efficient
    sorted_windows = np.sort(windows, axis=1)
    
    # Count distinct: number of unique values in each row
    # Unique values = 1 + number of indices where sorted[i] != sorted[i-1]
    # Use np.diff
    diffs = np.diff(sorted_windows, axis=1)
    # diffs has shape (num_windows, WINDOW-1)
    # distinct_count = 1 + sum(diffs != 0, axis=1)
    distinct_counts = 1 + np.sum(diffs != 0, axis=1)
    
    # Find the index with minimum distinct count
    min_idx = np.argmin(distinct_counts)
    best_n = N_MIN + min_idx
    best_count = distinct_counts[min_idx]
    
    return best_n
