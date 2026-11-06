import time
import numpy as np

N = 17
MIN_DIST = 6
MAX_WORD = 1 << N


def construct():
    start_time = time.time()

    # Precompute popcounts using a lookup table for speed
    # We only need to identify bad_diffs (weight < MIN_DIST)
    # Using a fast method to generate bad_diffs
    bad_diffs = []
    for d in range(MAX_WORD):
        if bin(d).count('1') < MIN_DIST:
            bad_diffs.append(d)
    
    bad_diffs_np = np.array(bad_diffs, dtype=np.uint32)
    # Note: bad_diffs_np contains indices of words at distance < MIN_DIST from 0.
    # For any u, the conflicting words are u ^ bad_diffs_np.

    best_code = []
    best_size = 0
    max_restarts = 1000 # Increased restarts, each is faster due to vectorization
    
    # Sample size for MRV heuristic
    # Larger sample = better choice, but more memory/time per step
    # 100 samples * 26000 bad_diffs = 2.6M operations per step
    # With ~250 steps, that's 650M ops per restart. 
    # This might be too slow for 1000 restarts in 60s.
    # Let's estimate: 650M simple numpy ops might take ~1-2 seconds.
    # So ~30-60 restarts in 60s.
    # Let's try sample size 50 to be safe, allowing more restarts.
    sample_size = 50

    for restart in range(max_restarts):
        if time.time() - start_time > 55:
            break

        is_avail = np.ones(MAX_WORD, dtype=bool)
        code = []

        while True:
            if time.time() - start_time > 55.5:
                break

            avail_idx = np.nonzero(is_avail)[0]
            if len(avail_idx) == 0:
                break

            # If few candidates left, check all
            if len(avail_idx) <= sample_size:
                samples = avail_idx
            else:
                samples = np.random.choice(avail_idx, sample_size, replace=False)
            
            # Vectorized degree calculation
            # samples shape: (S,)
            # bad_diffs_np shape: (B,)
            # neighbors_2d shape: (S, B)
            neighbors_2d = samples[:, None] ^ bad_diffs_np[None, :]
            
            # Get availability of all these neighbors
            # is_avail is 1D bool array
            # is_avail[neighbors_2d] gives (S, B) bool array
            degs = is_avail[neighbors_2d].sum(axis=1)
            
            # Find the sample with minimum degree
            min_idx = np.argmin(degs)
            best_u = int(samples[min_idx])
            
            code.append(best_u)
            
            # Mark best_u and its neighbors as unavailable
            # We can reuse the row from neighbors_2d for best_u
            # Or just recompute: neighbors = best_u ^ bad_diffs_np
            neighbors = best_u ^ bad_diffs_np
            is_avail[neighbors] = False

        if len(code) > best_size:
            best_size = len(code)
            best_code = code.copy()

    return best_code
