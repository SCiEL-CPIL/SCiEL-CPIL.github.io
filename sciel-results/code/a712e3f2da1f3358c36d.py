import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Leave 5s margin
    
    # 1. Generate the [17, 8, 6] linear code
    G = [
        0b10000000111100000,
        0b01000000110011000,
        0b00100000101010100,
        0b00010000011001100,
        0b00001000010101010,
        0b00000100001100110,
        0b00000010001010101,
        0b00000001000110101
    ]
    
    code = []
    for mask in range(256):
        word = 0
        for i in range(8):
            if (mask >> i) & 1:
                word ^= G[i]
        code.append(word)
        
    # Verify linearity and distance
    # If this code is invalid, fall back to seed
    # Quick check: min dist >= 6
    min_d = 17
    for i in range(len(code)):
        wi = code[i]
        for j in range(i+1, len(code)):
            d = (wi ^ code[j]).bit_count()
            if d < min_d:
                min_d = d
            if min_d < 6:
                # Fallback to simple seed
                seed_words = []
                for slope in range(16):
                    for intercept in range(2):
                        word = 0
                        for point in range(16):
                            bit = ((slope & point).bit_count() + intercept) % 2
                            word |= bit << point
                        seed_words.append(word)
                return seed_words

    best_code = code.copy()
    best_size = len(code)
    
    # Pre-compute all vectors
    all_vectors = np.arange(2**17, dtype=np.uint32)
    
    def calc_top2(c_list):
        """
        Calculates min1_dist, min1_idx, min2_dist, min2_idx for all 2^17 vectors
        relative to the code c_list.
        """
        c_arr = np.array(c_list, dtype=np.uint32)
        # Reshape to (131072, 1) and (1, M) for broadcasting
        v_col = all_vectors[:, None]
        c_row = c_arr[None, :]
        
        xor = v_col ^ c_row
        
        # Convert uint32 to uint8 for unpackbits
        # Each uint32 is 4 bytes
        xor_bytes = xor.view(np.uint8).reshape(-1, 4)
        
        # unpackbits along the last axis
        # Shape: (131072, M, 32)
        bits = np.unpackbits(xor_bytes, axis=1)
        
        # Sum to get Hamming distance
        # Shape: (131072, M)
        dists = bits.sum(axis=2).astype(np.int16)
        
        # Find min 1
        min1_dist = dists.min(axis=1)
        min1_idx = dists.argmin(axis=1)
        
        # Find min 2
        dists_masked = dists.copy()
        # Mask out the min 1
        indices = np.arange(len(c_list)) # Wait, indices for rows
        rows = np.arange(dists.shape[0])
        dists_masked[rows, min1_idx] = 17 # Max possible distance is 17
        
        min2_dist = dists_masked.min(axis=1)
        min2_idx = dists_masked.argmin(axis=1)
        
        available = np.where(min1_dist >= 6)[0]
        
        return min1_dist, min1_idx, min2_dist, min2_idx, available

    # Initial calculation
    min1_dist, min1_idx, min2_dist, min2_idx, available = calc_top2(code)
    
    # Check if we can add words immediately
    while len(available) > 0 and time.time() < deadline:
        # Pick a random available vector
        idx = np.random.choice(available)
        v = int(all_vectors[idx])
        code.append(v)
        best_code = code.copy()
        best_size = len(code)
        
        # Recalculate
        min1_dist, min1_idx, min2_dist, min2_idx, available = calc_top2(code)
        
    # If stuck, try swaps
    # We will try to remove a word that blocks many vectors
    # and add a vector that becomes available
    
    # To avoid infinite loops or getting stuck in local minima,
    # we track the number of swaps tried for each configuration
    # But given time limit, we just iterate.
    
    swaps_tried = 0
    max_swaps = 50 # Rough estimate of how many full recalcs we can afford
    
    while time.time() < deadline and swaps_tried < max_swaps:
        if len(available) > 0:
            # Add another word if possible
            idx = np.random.choice(available)
            v = int(all_vectors[idx])
            code.append(v)
            best_code = code.copy()
            best_size = len(code)
            min1_dist, min1_idx, min2_dist, min2_idx, available = calc_top2(code)
            continue
            
        # No words available. Try a swap.
        # Count how many vectors each codeword is the min1 for
        counts = np.bincount(min1_idx, minlength=len(code))
        
        # Pick the most blocking codeword
        # Add some randomness to escape local minima
        top_k = 10
        top_indices = np.argsort(counts)[-top_k:]
        c_idx_to_remove = np.random.choice(top_indices)
        
        # Remove it
        removed_word = code.pop(c_idx_to_remove)
        
        # Fast update of min1 using min2
        # For all vectors where min1_idx was c_idx_to_remove, min1 becomes min2
        mask = (min1_idx == c_idx_to_remove)
        min1_dist[mask] = min2_dist[mask]
        min1_idx[mask] = min2_idx[mask]
        
        available = np.where(min1_dist >= 6)[0]
        
        if len(available) > 0:
            # We can add a word
            # Pick one with max min1_dist to be "safest"
            # Or random
            idx = np.random.choice(available)
            v = int(all_vectors[idx])
            code.append(v)
            
            # Now code size is same as before swap.
            # But we need to check if we can add MORE.
            # So we must recalculate top2 for the new configuration.
            min1_dist, min1_idx, min2_dist, min2_idx, available = calc_top2(code)
            
            # If we can add more, great.
            # If not, we are at same size.
            # We keep this configuration as it might be better for future swaps.
            # But we should check if we improved best_size?
            # No, size is same.
            
            swaps_tried += 1
        else:
            # Could not add a word. Revert the swap.
            code.insert(c_idx_to_remove, removed_word)
            # Recalculate top2 to restore state?
            # Or just update back?
            # Updating back is tricky because min2 might have been from the removed word?
            # No, min2 is the second smallest. If min1 was removed, min2 is correct.
            # If we revert, min1 goes back to removed word.
            # We can just recalculate to be safe, or trust the fast update reverse.
            # Reverse fast update:
            # For all vectors where min1_idx is now min2_idx (from the swap),
            # we need to restore min1 to removed_word?
            # This is error-prone. Just recalculate.
            min1_dist, min1_idx, min2_dist, min2_idx, available = calc_top2(code)
            swaps_tried += 1

    # Final check: ensure best_code is valid
    # And return it
    return best_code
