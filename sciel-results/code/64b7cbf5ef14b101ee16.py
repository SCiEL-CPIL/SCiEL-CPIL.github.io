import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 84.0  # Safety margin for 90s limit
    
    N = 18
    MIN_D = 6
    TARGET_SIZE = 512
    
    # Vectorized Greedy Search
    # We maintain a list of valid words.
    # We generate batches of random candidates and check them against all valid words.
    
    valid_words = [0]
    
    # Pre-allocate a large array for candidates if needed, or just use lists
    # Using lists for valid_words is fine for 512 items.
    
    while len(valid_words) < TARGET_SIZE and time.time() < deadline - 2.0:
        # Generate a batch of candidates
        # The number of candidates per batch can be adjusted.
        # 1000 candidates per batch is a good start.
        batch_size = 2000
        candidates = np.random.randint(0, 1 << N, size=batch_size)
        
        # Convert candidates to a 2D array of bits for XOR?
        # Or just use integer XOR and bit_count.
        # NumPy doesn't have a vectorized bit_count for integers directly in all versions,
        # but we can use a lookup table or loop if necessary.
        # However, (a ^ b).bit_count() is fast in Python 3.10+.
        # To vectorize, we can convert to bits.
        
        # Let's try a hybrid: 
        # If valid_words is small, Python loop is okay.
        # If valid_words is large, we need speed.
        
        # Strategy:
        # 1. For each candidate, check against all valid words.
        # 2. If we use Python loops, 2000 candidates * 512 words = 1M ops.
        #    1M bit_counts in Python might take ~0.5s.
        #    We have 84s. So we can do ~160 batches.
        #    160 * 2000 = 320,000 candidates.
        #    Probability of finding a valid word?
        #    Density of valid words drops as we add more.
        
        # Let's optimize the inner loop.
        # We can store valid_words as a numpy array of ints.
        # And use a list comprehension or generator.
        
        vw_arr = np.array(valid_words)
        
        new_valids = []
        
        # Vectorized check using broadcasting is hard with bit_count.
        # Let's use a lookup table for popcount of 18-bit numbers?
        # 2^18 = 262144 entries. 
        # popcount_table = np.array([i.bit_count() for i in range(1<<18)])
        # This takes ~10ms to build.
        
        if not hasattr(construct, 'popcount_table'):
            construct.popcount_table = np.array([i.bit_count() for i in range(1 << N)], dtype=np.int8)
        
        pc_table = construct.popcount_table
        
        # For each candidate in the batch:
        # We need to check if min distance to vw_arr is >= 6.
        # dist = pc_table[candidate ^ vw_arr]
        # min_dist = np.min(dist)
        
        # This can be done in a loop over candidates, but vectorized over valid_words.
        # 2000 candidates.
        # for c in candidates:
        #    dists = pc_table[c ^ vw_arr]
        #    if np.min(dists) >= 6:
        #        new_valids.append(c)
        
        # This is 2000 * (vectorized op of size 512).
        # This is fast.
        
        for c in candidates:
            # Check against all current valid words
            # If c is already in valid_words, skip? 
            # Random collision is unlikely, but possible.
            # We can just check distance.
            
            # Optimization: Check against the last few added words first?
            # Or just all.
            
            dists = pc_table[c ^ vw_arr]
            if np.min(dists) >= MIN_D:
                new_valids.append(c)
                # Update vw_arr? 
                # If we add to vw_arr, the next candidate in the batch must be checked against it.
                # This makes it sequential.
                # To keep it vectorized, we can check against the *original* batch of valid words,
                # and then check new_valids against each other?
                # No, if we add A and B, and A and B are close, we might add both.
                # We must ensure new_valids are mutually distant.
                
                # Simpler: Add to a temporary list, and check against existing + new.
                # This is O(k^2) for the batch.
                
                # Given the small size (512), sequential addition is fine.
                vw_arr = np.append(vw_arr, c)
                valid_words.append(c)
                
                if len(valid_words) >= TARGET_SIZE:
                    break
        
        # If we didn't add enough, we might need to increase batch size or change strategy.
        # But 2000 per iteration is fast.
        
    return valid_words
