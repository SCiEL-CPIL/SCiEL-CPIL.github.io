import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 58s safety margin for 60s limit

    N = 17
    MIN_D = 6
    TOTAL = 1 << N

    # Precompute popcount for all 17-bit values using a fast method
    # Method: use numpy's bincount or a lookup table
    # For 131072 values, a simple list comprehension with bin().count('1') is slow.
    # Let's use a smarter approach:
    # popcount(x) for x < 2^17 can be computed via lookup of low 8 bits + high 9 bits?
    # Or just use np.unpackbits on an array?
    # Actually, the fastest in pure Python/NumPy for this size might be:
    # Create an array of 0..TOTAL-1, then use a vectorized popcount.
    # NumPy doesn't have a direct popcount, but we can use:
    # np.sum(np.unpackbits(arr.astype(np.uint8).reshape(-1, 1) << 0, axis=1), axis=1)
    # But unpackbits works on uint8. We need to handle 17 bits.
    # Let's do it in two steps: low 8 bits and high 9 bits.
    
    # Precompute popcount for 0..255
    pc8 = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        pc8[i] = bin(i).count('1')
    
    # Now for any x in [0, 2^17):
    # pc(x) = pc8[x & 0xFF] + pc8[(x >> 8) & 0xFF]  -- wait, x>>8 can be up to 2^9-1 = 511
    # So we need pc for 0..511.
    pc9 = np.zeros(512, dtype=np.uint8)
    for i in range(512):
        pc9[i] = bin(i).count('1')
    
    def fast_popcount_array(arr):
        """Vectorized popcount for array of uint32 values < 2^17."""
        arr = arr.astype(np.uint32)
        low = (arr & 0xFF).astype(np.uint8)
        high = ((arr >> 8) & 0x1FF).astype(np.uint8)
        return (pc8[low] + pc9[high]).astype(np.uint8)
    
    # For a single value, use a lookup table
    # Let's precompute for all 2^17 values once.
    # 131072 * 2 lookups is fine.
    all_vals = np.arange(TOTAL, dtype=np.uint32)
    popcount_lookup = fast_popcount_array(all_vals)
    
    # 1. Generate the known 256-word linear code [17,8,6]
    # We'll use a vectorized greedy approach to find 8 basis vectors.
    
    def generate_linear_code():
        code = np.array([0], dtype=np.uint32)
        basis = []
        
        for _ in range(8):
            if time.time() > deadline - 40:
                break
            # Check all candidates not in code
            # We want min distance to code >= 6
            # Vectorized: compute distance from all candidates to all code words
            # But we only need to find ONE valid g.
            # Let's iterate through candidates in a smart order.
            # Actually, let's just check all 131072 candidates at once.
            
            # Distances: candidates (TOTAL,) x code (K,)
            # This is 131072 * K. For K=256, it's 33M. OK.
            # But we can be smarter: we only need to check candidates not in code.
            # Let's just check all.
            
            # Compute min distance from each candidate to the code
            # dist[i, j] = popcount(candidates[i] ^ code[j])
            # We want min_j dist[i, j] >= 6
            
            # To save memory, we can do this in chunks if needed, but 33M uint8 is 33MB. OK.
            
            # Actually, we can optimize: we only need to find if ANY candidate has min_dist >= 6.
            # And we want to pick one.
            
            # Let's compute distances in a memory-efficient way.
            # For each candidate i, min_dist = min(popcount(i ^ c) for c in code)
            
            # Vectorized approach:
            # candidates[:, None] ^ code[None, :] -> (TOTAL, K)
            # Then popcount of that.
            
            # This creates a large array. Let's do it in chunks of 10000 candidates.
            found = False
            chunk_size = 10000
            for start in range(0, TOTAL, chunk_size):
                if time.time() > deadline - 40:
                    return code.tolist()
                end = min(start + chunk_size, TOTAL)
                candidates = np.arange(start, end, dtype=np.uint32)
                
                # Compute distances
                dists = popcount_lookup[candidates[:, None] ^ code[None, :]]
                min_dists = np.min(dists, axis=1)
                
                valid = np.where(min_dists >= MIN_D)[0]
                if len(valid) > 0:
                    # Pick the first valid candidate
                    g = candidates[valid[0]]
                    basis.append(g)
                    # Update code
                    code = np.concatenate([code, code ^ g])
                    found = True
                    break
            
            if not found:
                break
                
        return code.tolist()

    best_code = generate_linear_code()
    best_size = len(best_code)
    best_set = set(best_code)
    
    # Now, try to extend the code beyond 256 words.
    # Strategy: Local search with add/swap operations.
    
    # Precompute a list of all words not in the code
    all_words = list(range(TOTAL))
    
    def get_candidates(code_set):
        return [w for w in all_words if w not in code_set]
    
    # Vectorized check: which candidates can be added?
    def can_add_vectorized(candidates, code_np, code_set):
        if len(candidates) == 0:
            return np.array([], dtype=bool)
        
        candidates_np = np.array(candidates, dtype=np.uint32)
        
        # Compute min distance to code
        # To save memory, do in chunks
        valid_mask = np.zeros(len(candidates_np), dtype=bool)
        chunk_size = 5000
        for start in range(0, len(candidates_np), chunk_size):
            end = min(start + chunk_size, len(candidates_np))
            chunk = candidates_np[start:end]
            
            dists = popcount_lookup[chunk[:, None] ^ code_np[None, :]]
            min_dists = np.min(dists, axis=1)
            
            valid_mask[start:end] = min_dists >= MIN_D
        
        # Filter out those already in code_set
        final_mask = np.zeros(len(candidates_np), dtype=bool)
        for i in np.where(valid_mask)[0]:
            w = int(candidates_np[i])
            if w not in code_set:
                final_mask[i] = True
                
        return final_mask

    # Local improvement loop
    def local_improve(code, code_set, time_limit):
        code_np = np.array(code, dtype=np.uint32)
        candidates = get_candidates(code_set)
        
        if len(candidates) == 0:
            return code, code_set
        
        # First, try to add as many as possible
        while time.time() < time_limit:
            # Get all addable candidates
            mask = can_add_vectorized(candidates, code_np, code_set)
            
            if not np.any(mask):
                break
                
            new_words = np.array(candidates, dtype=np.uint32)[mask]
            code.extend(new_words.tolist())
            code_set.update(new_words.tolist())
            code_np = np.array(code, dtype=np.uint32)
            
            # Update candidates (remove added ones)
            new_set = set(new_words.tolist())
            candidates = [w for w in candidates if w not in new_set]
            
            if len(candidates) == 0:
                break
        
        # Now try swaps
        while time.time() < time_limit:
            if len(code) == 0:
                break
                
            # Pick a random word to remove
            idx = random.randint(0, len(code) - 1)
            removed = code[idx]
            code.pop(idx)
            code_set.remove(removed)
            code_np = np.array(code, dtype=np.uint32)
            
            # Try to add words
            mask = can_add_vectorized(candidates, code_np, code_set)
            
            added = 0
            if np.any(mask):
                new_words = np.array(candidates, dtype=np.uint32)[mask]
                # Add at most 1 word to keep it a swap (or more if beneficial)
                # For now, add the first one
                w = int(new_words[0])
                code.append(w)
                code_set.add(w)
                code_np = np.array(code, dtype=np.uint32)
                candidates.remove(w)
                added = 1
            
            if added == 0:
                # Revert
                code.append(removed)
                code_set.add(removed)
                code_np = np.array(code, dtype=np.uint32)
            else:
                # Check if we improved size
                # If we swapped 1 for 1, size is same.
                # If we added >1, size increased.
                # For now, just continue searching.
                pass
        
        return code, code_set

    # Main search loop
    while time.time() < deadline - 5:
        # Try local improvement
        improved_code, improved_set = local_improve(best_code.copy(), best_set.copy(), deadline - 10)
        
        if len(improved_code) > best_size:
            best_code = improved_code
            best_set = improved_set
            best_size = len(best_code)
        else:
            # If no improvement, try a random restart or different strategy
            # For now, just shuffle and try again
            random.shuffle(best_code)
            best_set = set(best_code)
            
        # Check time
        if time.time() > deadline - 15:
            break

    return best_code
