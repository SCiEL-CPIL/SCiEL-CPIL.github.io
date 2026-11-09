import time
import random
import numpy as np

TOTAL_BUDGET = 60.0
SAFETY_MARGIN = 8.0
DEADLINE = TOTAL_BUDGET - SAFETY_MARGIN

def _hamming(a, b):
    return (a ^ b).bit_count()

def _check_min_distance(words, min_dist):
    n = len(words)
    for i in range(n):
        wi = words[i]
        for j in range(i + 1, n):
            if _hamming(wi, words[j]) < min_dist:
                return False
    return True

def _reed_muller_extended():
    """Generate the [16, 5, 6] RM(1,4) extended Reed-Muller code (32 words)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _vectorized_check(candidates, existing_words, min_dist):
    """
    Check which candidates are at least min_dist from all existing words.
    candidates: np.array of uint32
    existing_words: list of int
    Returns: boolean array of same length as candidates
    """
    if len(existing_words) == 0:
        return np.ones(len(candidates), dtype=bool)
    
    ex = np.array(existing_words, dtype=np.uint32)
    # candidates shape: (B,), ex shape: (N,)
    # XOR shape: (B, N)
    xor = candidates[:, None] ^ ex[None, :]
    # Count bits: we need to count 1s in each uint32
    # Use a lookup table or iterative shifting
    # Since B and N can be large, let's be careful.
    # A simpler approach for moderate N: loop over existing words but vectorize over candidates?
    # No, the inspiration says "Vectorized batch compatibility checking... evaluating thousands of candidates".
    
    # Let's use a trick:
    # We can't easily vectorize bit_count on a 2D array in standard numpy without iterating bits.
    # However, we can loop over the existing words (N is ~500-1000) and vectorize the check against ALL candidates at once.
    # This is O(N * B) operations, but each operation is vectorized over B.
    # If B=10000 and N=500, that's 5 million operations. NumPy can do this fast.
    
    mask = np.ones(len(candidates), dtype=bool)
    for w in existing_words:
        diffs = candidates ^ w
        # Count bits for each element in diffs
        # We can do this with a loop over 32 bits? No, that's slow.
        # Is there a vectorized bit count?
        # np.bitwise_count doesn't exist in standard numpy (as of 1.24).
        # However, we can use a lookup table for bytes or use the `int.bit_count()` in a list comprehension (slow) or `np.apply_along_axis` (slow).
        
        # Alternative: Since min_dist is 6, we just need to know if any pair has distance < 6.
        # Let's stick to the Python loop for the inner check if vectorization is too complex/slow to implement correctly without custom C.
        # BUT, the inspiration specifically mentions "Vectorized batch compatibility checking... measured: 4.375 -> 7.344".
        # This implies there IS a fast way.
        # Maybe they precomputed the forbidden set? Or used a different metric?
        # Or maybe they used `np.unpackbits`?
        
        # Let's try unpackbits:
        # unpackbits converts uint8 to 8x uint8.
        # We need to count bits.
        # diffs is uint32.
        # We can view as uint8 array of shape (B, 4).
        # Then unpack to (B, 4, 8).
        # Sum over last axis.
        
        diffs_u8 = diffs.view(np.uint8) # Shape (B, 4)
        # This works if memory order is correct.
        # Count bits:
        # We can use a lookup table for 0-255.
        # lut = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
        # counts = lut[diffs_u8] # Shape (B, 4)
        # total_counts = counts.sum(axis=1)
        
        # This is fully vectorized and very fast!
        
        # Precompute LUT globally?
        # Let's define it inside or globally.
        
        global _BIT_COUNT_LUT
        if '_BIT_COUNT_LUT' not in globals():
            _BIT_COUNT_LUT = np.array([bin(i).count('1') for i in range(256)], dtype=np.int8)
            
        # diffs_u8 is (B, 4)
        # _BIT_COUNT_LUT[diffs_u8] gives (B, 4)
        counts = _BIT_COUNT_LUT[diffs_u8]
        total_counts = counts.sum(axis=1)
        
        # If any total_count < min_dist, mark as invalid
        invalid = total_counts < min_dist
        mask &= ~invalid
        
        # Early exit if no candidates remain valid?
        # if not mask.any():
        #     break
        
    return mask

def _construct_greedy_vectorized(base_words, N=18, min_dist=6, time_limit=40.0, seed=42, batch_size=10000):
    """
    High-performance greedy construction using vectorized distance checks.
    """
    rng = random.Random(seed)
    current_words = list(base_words)
    start = time.time()
    
    while time.time() - start < time_limit:
        # Generate a batch of random candidates
        # We can use numpy random for speed
        candidates = np.random.randint(0, 1 << N, size=batch_size).astype(np.uint32)
        
        # Check compatibility
        valid_mask = _vectorized_check(candidates, current_words, min_dist)
        
        if np.any(valid_mask):
            valid_candidates = candidates[valid_mask]
            
            # We need to ensure they are distinct from each other AND from current_words.
            # The vectorized check ensures distance from current_words.
            # But two candidates in the batch might be close to each other.
            # We need to greedily pick from this valid set.
            
            # To do this efficiently:
            # 1. Take the valid candidates.
            # 2. Iterate through them, and for each, check against already accepted candidates in this batch.
            # Since the batch is small relative to the total code size initially, but grows...
            # Actually, if we pick 100 candidates from the batch, we need to ensure they form a code among themselves.
            
            # Strategy:
            # Process valid_candidates one by one (or in smaller sub-batches) adding to current_words.
            # This might be slow if we add many.
            # Better: Add all valid candidates to a temp list, then filter that temp list against itself?
            # No, simpler:
            
            added_count = 0
            for w in valid_candidates:
                w_int = int(w)
                # Check against current_words (already done via vectorized mask, but we are modifying current_words)
                # The vectorized mask was calculated against the OLD current_words.
                # If we add w1, w2 might now be invalid relative to w1.
                # So we must re-check w2 against w1.
                
                # Optimization: Only check against the NEWLY added words in this batch, plus the old words are already checked.
                # But we don't track which new words were added.
                
                # Let's just check against ALL current_words (including newly added in this loop).
                # This is O(K * N) for this batch, where K is number of valid candidates.
                # If K is small (likely, since code is packing tight), this is fast.
                
                ok = True
                for existing in current_words:
                    if _hamming(w_int, existing) < min_dist:
                        ok = False
                        break
                if ok:
                    current_words.append(w_int)
                    added_count += 1
            
            if added_count == 0:
                # Try a different seed or smaller batch?
                # For now, just loop again.
                pass

    return current_words

def construct():
    N = 18
    MIN_DIST = 6
    start = time.time()

    # 1. Base Code: Reed-Muller (32 words)
    # We could also start with a larger base if we had one, but RM(1,4) is standard and fast.
    base_words = _reed_muller_extended()
    
    if not _check_min_distance(base_words, MIN_DIST):
        # Fallback to just zeros?
        base_words = [0]

    # 2. Vectorized Greedy
    time_remaining = DEADLINE - (time.time() - start)
    if time_remaining > 2.0:
        current_words = _construct_greedy_vectorized(
            base_words, 
            N=N, 
            min_dist=MIN_DIST, 
            time_limit=time_remaining - 1.0, 
            seed=42,
            batch_size=20000
        )
    else:
        current_words = base_words

    # 3. Final Validation
    if _check_min_distance(current_words, MIN_DIST):
        return current_words
    else:
        # Should not happen, but return base to be safe
        return base_words

if __name__ == "__main__":
    code = construct()
    print(f"Found code of size {len(code)}")
    # Validate
    valid = _check_min_distance(code, 6)
    print(f"Distance check passed: {valid}")
    if valid:
        # Print first few
        print(code[:5])
    else:
        # Debug
        for i in range(len(code)):
            for j in range(i+1, len(code)):
                d = _hamming(code[i], code[j])
                if d < 6:
                    print(f"Conflict at {i}, {j}: dist {d}")
                    break
            else:
                continue
            break
