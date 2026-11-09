"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    N = 18
    D = 6
    
    # We will use a greedy + local search approach.
    # The key insight is that A_2(18, 6) is likely larger than 32.
    # The extended quadratic residue code of length 18 (from length 16 QR code extended?) 
    # Wait, QR codes are usually length p^2. 
    # Let's look at known bounds for A_2(18, 6).
    # A_2(16, 6) = 32 (RM(1,4) extended? No, RM(1,4) has length 16, dist 8, size 32. 
    # Wait, RM(1,4) has min dist 2^{4-1}=8. So it is a (16, 32, 8) code.
    # If we just take RM(1,4), we have 32 words with dist 8.
    # Can we add more words?
    # The task mentions a 512-word extended quadratic-residue construction.
    # Let's verify if 512 is feasible.
    # If A_2(18, 6) is 512, that's a huge jump from 32.
    # Let's try to construct a code greedily and see how far we get.
    
    # Strategy:
    # 1. Start with a good base code (e.g., RM(1,4) padded to 18 bits, or random).
    # 2. Greedily add words that are at distance >= 6 from all existing words.
    # 3. Local search: try to swap out a word for a new one that allows more additions?
    #    Or simply iterate: try to add a random word, if it fits, add it.
    #    Repeat until no more can be added.
    # 4. To escape local optima, we can do multiple restarts with different random seeds/orders.
    
    # Let's use NumPy for fast distance checks.
    # Represent words as a list of ints.
    # To check if a new word `w` is compatible with a set `S`:
    #   for each s in S: popcount(w ^ s) >= 6
    
    # Optimization:
    # Precompute popcounts for all 2^18 integers? No, 2^18 = 262144. That's small.
    # We can precompute a lookup table for popcount of 18-bit integers.
    # Then popcount(x) = table[x].
    # Even better, we can just use int.bit_count() which is fast in Python 3.10+.
    
    # Let's try a simple greedy first.
    # 1. Generate all 2^18 integers? No, that's 262k.
    # 2. Shuffle them?
    # 3. Iterate through the shuffled list, adding if compatible.
    
    # But 262k * (size of code) checks might be slow if code size is large.
    # If code size is ~100, 262k * 100 = 26M operations. In Python, this is ~10-20s.
    # We have 85s. So we can do a few restarts.
    
    # Let's implement a faster check using NumPy.
    # Keep the current code as a NumPy array of uint16 (or int32).
    # For a new candidate `w`, compute `code ^ w` for all in code, then bit_count.
    # NumPy doesn't have a direct bit_count for arrays in older versions, 
    # but we can use `np.unpackbits` or just loop if N is small.
    # Actually, `int.bit_count()` is C-speed. A Python loop over 100 items is fast.
    # A Python loop over 262k candidates * 100 checks = 26M bit_counts.
    # 26M simple operations in Python might take ~5-10 seconds.
    # So we can do ~5-10 full greedy passes.
    
    # Let's refine the greedy:
    # Instead of random order, we can use a "best-fit" or "worst-fit" heuristic?
    # Or just random restarts.
    
    # Let's also try to start with the RM(1,4) code (32 words) as a seed.
    # This might lead to a denser packing.
    
    def rm14_code():
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                # Extend to 18 bits by adding 2 zero bits? 
                # Or just use the 16-bit word as is (padded with 0s).
                words.append(word)
        return words

    best_code = []
    
    # Pre-generate all candidates? 
    # All 2^18 integers.
    # We can store them in a list.
    all_candidates = list(range(2**18))
    
    # We will perform multiple restarts.
    # Each restart:
    # 1. Initialize code with a random subset or RM code.
    # 2. Shuffle candidates (excluding those in initial code).
    # 3. Greedily add compatible candidates.
    # 4. Record size.
    
    # To speed up, we can filter candidates early.
    # If we have a set of codewords, we can maintain a "blocked" set?
    # No, because adding a new word blocks new things, but doesn't unblock.
    # So we just check compatibility.
    
    # Let's try a hybrid:
    # Start with RM(1,4).
    # Then greedily add from random order.
    
    # Let's also consider that the optimal code might be nonlinear.
    # Greedy random restarts is a good heuristic for finding large independent sets.
    
    num_restarts = 50  # We will adjust based on time
    rng = random.Random(42)
    
    for i in range(num_restarts):
        if time.time() > DEADLINE:
            break
            
        # Initialize code
        if i == 0:
            code = rm14_code()
        else:
            # Random initialization: pick 10 random words, ensure they are pairwise dist >= 6
            code = []
            attempts = 0
            while len(code) < 10 and attempts < 1000:
                w = rng.randint(0, 2**18 - 1)
                ok = True
                for c in code:
                    if (w ^ c).bit_count() < D:
                        ok = False
                        break
                if ok:
                    code.append(w)
                attempts += 1
        
        # Create a list of candidates to try
        # We can just iterate through all 2^18, but that's 262k.
        # Shuffling 262k items is fast.
        candidates = all_candidates[:]
        rng.shuffle(candidates)
        
        # Greedy addition
        # We need to check each candidate against all current codewords.
        # To make this faster, we can break early if a candidate is not compatible.
        
        # Optimization: 
        # If the code size is small, checking is fast.
        # As code grows, checking gets slower.
        # But we only need to check until we find a conflict.
        
        for w in candidates:
            if time.time() > DEADLINE:
                break
                
            # Check if w is already in code? 
            # Since we iterate through all, and code is a set/list, 
            # we can skip if w is in code. 
            # But checking `w in code` is O(N). 
            # Better to maintain a set of codewords for O(1) lookup.
            # But we also need the list for distance checks.
            
            # Let's use a set for membership and list for distance.
            # Actually, since we are iterating through ALL candidates, 
            # and we add to code, we can just check distance.
            # If w is in code, distance is 0, which is < 6, so it will be rejected.
            # So we don't need explicit membership check for correctness, 
            # but it saves time if we skip known codewords.
            # However, checking distance against all codewords is O(|code|).
            
            compatible = True
            for c in code:
                if (w ^ c).bit_count() < D:
                    compatible = False
                    break
            
            if compatible:
                code.append(w)
        
        if len(code) > len(best_code):
            best_code = code[:]
            # print(f"Restart {i}: size {len(code)}") # Debug
    
    # After greedy, we can try local search?
    # Local search: try to remove a word and add two? 
    # Or simply: iterate through all candidates again, 
    # and if a candidate is compatible with all EXCEPT one, 
    # try swapping? This is complex.
    
    # Given the time limit, the greedy restarts might be sufficient.
    # The 512-word QR code suggests we might reach ~500.
    # Let's see if 50 restarts of 262k candidates is feasible.
    # 262k * 50 = 13M candidates.
    # Average code size ~100.
    # 13M * 100 = 1.3B bit_count operations.
    # This is likely too slow in Python.
    
    # We need to optimize the inner loop.
    # Using NumPy for batch checks?
    # If we have a NumPy array of candidates, and a NumPy array of code,
    # we can compute XOR and bit counts vectorized?
    # `np.bitwise_xor` is vectorized.
    # `np.bit_count` is not directly available for arrays in all NumPy versions.
    # We can use `np.unpackbits` and sum, but that's slow.
    # Alternatively, we can use a lookup table for 18-bit popcount.
    # `popcount_table = np.array([i.bit_count() for i in range(2**18)], dtype=np.uint8)`
    # Then `popcount_table[xor_array]` gives the popcounts.
    # This is vectorized and fast!
    
    # Let's rewrite the greedy using this lookup table.
    
    popcount_table = np.array([i.bit_count() for i in range(2**18)], dtype=np.uint8)
    
    best_code = []
    # Re-run with optimized inner loop
    for i in range(100): # More restarts possible now
        if time.time() > DEADLINE:
            break
            
        if i == 0:
            code = rm14_code()
        else:
            code = []
            attempts = 0
            while len(code) < 10 and attempts < 1000:
                w = rng.randint(0, 2**18 - 1)
                ok = True
                for c in code:
                    if (w ^ c).bit_count() < D:
                        ok = False
                        break
                if ok:
                    code.append(w)
                attempts += 1
        
        candidates = np.array(all_candidates, dtype=np.uint32)
        rng.shuffle(candidates)
        
        code_arr = np.array(code, dtype=np.uint32)
        
        # We will iterate through candidates one by one to avoid huge memory overhead
        # of checking all candidates at once against a growing code.
        # But we can batch check a chunk of candidates against the current code.
        
        # Let's process candidates in batches of 1000.
        for start in range(0, len(candidates), 1000):
            if time.time() > DEADLINE:
                break
                
            batch = candidates[start:start+1000]
            
            # Compute XOR of each candidate in batch with each codeword in code.
            # batch shape: (1000,)
            # code_arr shape: (K,)
            # We want a (1000, K) matrix of XORs.
            # This might be large if K is large.
            # If K=100, 1000*100 = 100k ints. Fine.
            # If K=500, 1000*500 = 500k ints. Fine.
            
            if len(code_arr) == 0:
                # Add all in batch
                code = list(batch)
                code_arr = batch
                continue
            
            # Reshape for broadcasting
            # batch[:, None] shape (1000, 1)
            # code_arr[None, :] shape (1, K)
            # xor_mat shape (1000, K)
            xor_mat = batch[:, None] ^ code_arr[None, :]
            
            # Get popcounts
            # popcount_table[xor_mat]
            # xor_mat values are in [0, 2^18)
            # popcount_table is indexed by these values.
            dists = popcount_table[xor_mat]
            
            # We need dists >= 6 for all K codewords.
            # min_dists = np.min(dists, axis=1)
            # valid = min_dists >= 6
            # But we need to be careful: if any dist < 6, invalid.
            # np.all(dists >= 6, axis=1)
            
            valid = np.all(dists >= D, axis=1)
            
            # Add valid candidates to code
            new_words = batch[valid]
            if len(new_words) > 0:
                code.extend(new_words.tolist())
                # Update code_arr for next batch?
                # We can just keep appending to code_arr.
                # But code_arr is a NumPy array. 
                # It's easier to just keep a list and convert to array when needed.
                # Or just append to the NumPy array? 
                # np.concatenate is slow if done repeatedly.
                # Let's just keep a list `code_list` and convert to array for batch checks.
                # Wait, I'm mixing list and array.
                # Let's stick to list for `code` and convert to array inside the loop.
                
                # To optimize, I'll restructure the loop to use a list and convert.
                pass
        
        # The above loop structure is getting messy. 
        # Let's simplify: 
        # Just iterate through candidates one by one, but use the lookup table for speed?
        # No, the lookup table is for NumPy arrays.
        # For a single integer, `w.bit_count()` is fast.
        # The bottleneck is the Python loop over candidates.
        # 262k iterations * 100 checks = 26M bit_counts.
        # In Python, 26M simple ops is ~1s.
        # So 100 restarts * 1s = 100s. Too slow.
        
        # We need to reduce the number of checks or restarts.
        # Or use the NumPy batch approach correctly.
        
        # Let's implement the NumPy batch approach properly.
        
        # Reset code for this restart
        if i == 0:
            code = rm14_code()
        else:
            code = []
            attempts = 0
            while len(code) < 10 and attempts < 1000:
                w = rng.randint(0, 2**18 - 1)
                ok = True
                for c in code:
                    if (w ^ c).bit_count() < D:
                        ok = False
                        break
                if ok:
                    code.append(w)
                attempts += 1
        
        candidates = np.array(all_candidates, dtype=np.uint32)
        rng.shuffle(candidates)
        
        code_list = code[:]
        
        # Process in batches
        BATCH_SIZE = 5000
        for start in range(0, len(candidates), BATCH_SIZE):
            if time.time() > DEADLINE:
                break
            
            batch = candidates[start:start+BATCH_SIZE]
            
            if len(code_list) == 0:
                code_list = batch.tolist()
                continue
            
            code_arr = np.array(code_list, dtype=np.uint32)
            
            # Check compatibility
            # We need to check each candidate in batch against all in code_list.
            # To avoid huge memory, we can sub-batch the code if it's very large.
            # But 5000 * 1000 = 5M ints = 20MB. Fine.
            
            xor_mat = batch[:, None] ^ code_arr[None, :]
            dists = popcount_table[xor_mat]
            valid = np.all(dists >= D, axis=1)
            
            new_words = batch[valid]
            if len(new_words) > 0:
                code_list.extend(new_words.tolist())
        
        if len(code_list) > len(best_code):
            best_code = code_list[:]
    
    if not best_code:
        # Fallback
        best_code = rm14_code()
        
    return best_code

# EVOLVE-BLOCK-END
