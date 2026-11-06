import numpy as np
import time

def construct():
    N = 17
    MIN_DIST = 6
    DEADLINE = 85.0  # seconds, safety margin before 90s kill
    start_time = time.time()
    
    max_val = 1 << N
    
    # Generate all 2^17 words as uint32
    all_words = np.arange(max_val, dtype=np.uint32)
    
    # Precompute bit counts for all possible XOR values (0 to 2^17 - 1)
    # popcount for each value
    popcounts = np.zeros(max_val, dtype=np.uint8)
    for i in range(1, max_val):
        popcounts[i] = popcounts[i >> 1] + (i & 1)
    
    # Greedy set packing with "fewest conflicts" heuristic
    # We'll maintain a set of selected words and a boolean array of "available" words
    available = np.ones(max_val, dtype=bool)
    selected = []
    
    # Precompute weights for sorting
    weights = popcounts[all_words]
    
    # Strategy: repeatedly pick the word with minimum "degree" (number of available neighbors within distance < MIN_DIST)
    # But computing degree for all is expensive. Instead, use a multi-pass greedy:
    # Pass 1: Sort by weight (ascending), pick greedily
    # Pass 2+: Try to add more words by looking at remaining available words
    
    # For efficiency, we'll use a chunked approach:
    # At each step, compute conflicts with selected words for a batch of candidates
    
    # Initialize: pick word 0 (all zeros)
    selected.append(0)
    available[0] = False
    
    # For fast conflict checking, maintain a list of selected words as a numpy array
    selected_arr = np.array(selected, dtype=np.uint32)
    
    # We'll process in batches to keep memory manageable
    BATCH_SIZE = 4096
    
    # Sort candidates by weight for initial greedy pass
    # But we want to be smart: use "fewest conflicts" heuristic
    # Approximation: words with moderate weight have fewer conflicts
    
    # Let's do a multi-stage approach:
    # Stage 1: Greedy by weight (low to high)
    # Stage 2: Try to fill gaps with remaining words
    
    # For the greedy pass, we'll iterate through words in order of increasing weight
    # and pick if compatible with all selected
    
    # To speed up, we'll check conflicts in batches
    order = np.argsort(weights, kind='stable')
    
    idx = 0
    while idx < max_val:
        if time.time() > DEADLINE:
            break
        
        # Take a batch of candidates
        batch_end = min(idx + BATCH_SIZE, max_val)
        batch = all_words[order[idx:batch_end]]
        
        # Filter to only available words
        batch_available_mask = available[batch]
        batch_candidates = batch[batch_available_mask]
        
        if len(batch_candidates) == 0:
            idx = batch_end
            continue
        
        # Check conflicts with selected words
        # For each candidate, compute min distance to any selected word
        # We need: for each candidate c, min over s in selected of popcount(c XOR s) >= MIN_DIST
        
        # Vectorized: for each candidate, compute XOR with all selected
        # This is (len(batch_candidates), len(selected_arr)) which can be large
        
        # Better: process in sub-batches if selected is large
        if len(selected_arr) < 50:
            # Direct comparison
            xor_matrix = batch_candidates[:, None] ^ selected_arr[None, :]
            dists = popcounts[xor_matrix]
            min_dists = np.min(dists, axis=1)
            valid_mask = min_dists >= MIN_DIST
        else:
            # Process selected in chunks
            min_dists = np.full(len(batch_candidates), 255, dtype=np.uint8)
            chunk_size = max(1, 1000000 // max(1, len(batch_candidates)))
            for s_start in range(0, len(selected_arr), chunk_size):
                s_end = min(s_start + chunk_size, len(selected_arr))
                xor_matrix = batch_candidates[:, None] ^ selected_arr[s_start:s_end][None, :]
                dists = popcounts[xor_matrix]
                min_dists = np.minimum(min_dists, np.min(dists, axis=1))
            valid_mask = min_dists >= MIN_DIST
        
        # Pick all valid candidates
        valid_candidates = batch_candidates[valid_mask]
        
        if len(valid_candidates) > 0:
            # Add them to selected
            selected.extend(valid_candidates.tolist())
            selected_arr = np.array(selected, dtype=np.uint32)
            available[valid_candidates] = False
            
            # Remove from available: also mark their neighbors as unavailable
            # Actually, we'll handle this in the next batch by re-checking
            # But for efficiency, let's pre-mark conflicts
            # For each newly added word, mark all words within distance < MIN_DIST as unavailable
            # This is expensive. Instead, we'll just rely on the distance check in subsequent batches.
            
            # To improve efficiency, let's proactively mark conflicts for the new words
            # For each new word, find all words within distance < MIN_DIST and mark them unavailable
            # This is O(2^17 * k) per new word which is too slow for many words.
            # Alternative: just let the distance check handle it.
        
        idx = batch_end
    
    # Stage 2: Try to add more words from remaining available
    # Re-scan available words, this time without weight ordering
    remaining = np.where(available)[0]
    if len(remaining) > 0 and time.time() < DEADLINE:
        # Shuffle for diversity
        np.random.shuffle(remaining)
        
        for word in remaining:
            if time.time() > DEADLINE:
                break
            
            # Check if compatible with all selected
            xor_vals = selected_arr ^ word
            dists = popcounts[xor_vals]
            if np.min(dists) >= MIN_DIST:
                selected.append(int(word))
                selected_arr = np.append(selected_arr, word)
                available[word] = False
    
    # Final validation
    if len(selected) < 2:
        # Fallback: return the seed code
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    return selected
