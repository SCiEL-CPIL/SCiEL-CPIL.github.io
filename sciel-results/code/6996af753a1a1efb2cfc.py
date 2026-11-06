import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Determine N based on memory
    # 50M int64 is 400MB.
    N = 50_000_000
    
    try:
        phi = np.arange(N + 1, dtype=np.int64)
    except MemoryError:
        N = 10_000_000
        try:
            phi = np.arange(N + 1, dtype=np.int64)
        except MemoryError:
            N = 1_000_000
            phi = np.arange(N + 1, dtype=np.int64)

    # Sieve
    N_valid = N
    for i in range(2, N + 1):
        if phi[i] == i:
            # i is prime
            phi[i::i] -= phi[i::i] // i
        
        # Check time every 1000 iterations to avoid overhead
        if i % 1000 == 0 and time.time() > deadline:
            N_valid = i - 1
            break
    else:
        N_valid = N

    # Scan for longest distinct run
    best_k = 1
    best_s = 1
    
    counts = {}
    left = 0  # 0-indexed index in phi array, corresponds to value 1
    
    # We scan indices 1 to N_valid
    # phi[0] is 0, we ignore it.
    for right in range(1, N_valid + 1):
        val = phi[right]
        counts[val] = counts.get(val, 0) + 1
        
        while counts[val] > 1:
            left_val = phi[left]
            counts[left_val] -= 1
            left += 1
            
        current_len = right - left
        # Length is right - left because left is 0-indexed start of window
        # Window is [left, right] inclusive.
        # Number of elements = right - left + 1?
        # If left=0, right=1, elements are phi[0], phi[1].
        # But we start at 1.
        # Let's trace:
        # right=1. left=0.
        # If distinct, len = 1 - 0 = 1? No.
        # Indices: 1. Count: 1.
        # left is the index of the first element in the window.
        # Initially left=0? No, we should start left=1?
        # Or treat left as the index *before* the start?
        
        # Let's use 1-based logic for clarity in the scan.
        # left_idx = 1
        # right_idx = 1
        # Window is [left_idx, right_idx]
        
    # Let's re-implement the scan with clear 1-based indexing.
    
    counts = {}
    left_idx = 1
    best_k = 0
    best_s = 1
    
    for right_idx in range(1, N_valid + 1):
        val = phi[right_idx]
        counts[val] = counts.get(val, 0) + 1
        
        while counts[val] > 1:
            left_val = phi[left_idx]
            counts[left_val] -= 1
            left_idx += 1
            
        current_len = right_idx - left_idx + 1
        if current_len > best_k:
            best_k = current_len
            best_s = left_idx
            
        if time.time() > deadline:
            break

    if best_k == 0:
        best_k = 1
        best_s = 1

    return {'start': best_s, 'length': best_k}
