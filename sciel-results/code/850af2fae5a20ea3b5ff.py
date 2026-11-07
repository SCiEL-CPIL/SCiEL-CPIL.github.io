import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # With NumPy, we can go much higher.
    # Memory: N * 8 bytes.
    # N = 10^8 -> 800 MB. This might be close to memory limits, but usually OK.
    # N = 5 * 10^7 -> 400 MB. Safer.
    N = 50_000_000
    
    try:
        sigma = np.ones(N + 1, dtype=np.int64)
        
        # Sieve using NumPy slicing
        for i in range(2, N + 1):
            # Only if i is not too large? No, all i contribute.
            # But for large i, the number of multiples is small.
            # The loop overhead in Python for 50 million iterations is the bottleneck now!
            # 50 million Python iterations for the outer loop is ~5-10 seconds.
            # The inner operation is fast.
            sigma[i::i] += i
            
    except MemoryError:
        # Fallback to smaller N
        N = 20_000_000
        sigma = np.ones(N + 1, dtype=np.int64)
        for i in range(2, N + 1):
            sigma[i::i] += i

    # Check time
    if time.time() > deadline - 10.0:
        # If sieve took too long, we might be tight on bucketing.
        pass

    # Bucketing
    # Converting NumPy array to list or iterating?
    # Iterating over NumPy array in Python is slower than list?
    # Let's stick to Python loop for bucketing, or vectorize?
    # M = k * sigma[k].
    # We need to bucket M.
    # We can compute M array:
    k_range = np.arange(1, N + 1, dtype=np.int64)
    M_vals = k_range * sigma[1:]  # sigma[0] is unused/1
    
    # Filter M <= 10^18
    # All M_vals for N=5*10^7:
    # Max M approx 5*10^7 * (5*10^7 * log(log(5*10^7))) ~ 2.5*10^15 * 3 ~ 7.5*10^15.
    # Well within 10^18.
    
    # Now we need to find collisions in M_vals.
    # We can use a dictionary.
    # But iterating 50 million items in Python to build a dict is slow (~5-10s).
    
    # Alternative: Use NumPy to find duplicates?
    # Hard to do directly.
    
    # Let's just iterate.
    buckets = defaultdict(list)
    
    # To speed up, we can use a list of tuples or just iterate.
    # 50 million iterations in Python might take 5-10 seconds.
    # Total time: Sieve (10s) + Bucketing (10s) = 20s. Very safe.
    
    # We can even increase N to 100,000,000.
    # Memory: 800MB.
    # Sieve: 100M outer loop iterations in Python. This is the bottleneck.
    # 100M Python loop iterations ~ 10-20 seconds.
    # Inner NumPy ops are fast.
    # Bucketing: 100M Python iterations ~ 10-20 seconds.
    # Total ~ 40s. Safe.
    
    # Let's try N = 80_000_000 to be safe on memory and time.
    
    # Re-run with N = 80_000_000?
    # I will set N = 60_000_000 as a good balance.
    
    # Actually, let's just use the N defined above (50M) and see.
    # If I want to be aggressive, I can set N = 100_000_000.
    # Let's stick to 50M for safety in this thought process, but in code I can push to 80M.
    
    N = 80_000_000
    try:
        sigma = np.ones(N + 1, dtype=np.int64)
        for i in range(2, N + 1):
            sigma[i::i] += i
    except MemoryError:
        N = 40_000_000
        sigma = np.ones(N + 1, dtype=np.int64)
        for i in range(2, N + 1):
            sigma[i::i] += i

    # Bucketing
    buckets = defaultdict(list)
    # Iterate
    # To make this faster, we can process in chunks? No, dict update is O(1).
    # Python loop is the limit.
    
    # We can break if time is up.
    # But we need the best M. If we break early, we might miss the best M.
    # However, collisions are rare. The best M is likely found early? No, it's random.
    # So we should process all.
    
    # If 80M takes too long, we return best so far.
    
    for k in range(1, N + 1):
        M = k * sigma[k]
        if M <= 10**18:
            buckets[M].append(k)
            
        if k % 1000000 == 0 and time.time() > deadline - 5.0:
            break
            
    best_M = 1
    best_ks = [1]
    best_count = 1
    
    for M, ks in buckets.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_M = M
            best_ks = ks
            
    return {'M': int(best_M), 'k': [int(k) for k in best_ks]}
