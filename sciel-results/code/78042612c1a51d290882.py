import time
import numpy as np

def construct():
    """
    Construct the largest set of 17-bit words with pairwise Hamming distance >= 6.
    Uses a structural change: precompute conflict graph via vectorized XOR/popcount,
    then apply greedy set packing with minimum-remaining-values heuristic.
    """
    start_time = time.time()
    deadline = start_time + 82.0  # 8s safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    TOTAL = 1 << N  # 131072
    
    # Precompute all words as uint32 array
    all_words = np.arange(TOTAL, dtype=np.uint32)
    
    # Extract bits into a (TOTAL, N) uint8 array
    bits = np.zeros((TOTAL, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = (all_words >> i) & 1
    
    # Compute weights for ordering
    weights = bits.sum(axis=1)
    
    # Strategy: 
    # 1. Start with a strong initial code (use RM(1,4) extended to 17 bits)
    # 2. Build a conflict adjacency structure
    # 3. Use greedy with MRV (minimum remaining values) to add more codewords
    
    # Initial code: RM(1,4) on first 16 bits, plus bit 16 = 0
    initial_code = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            initial_code.append(word)
    
    # Convert to indices
    initial_indices = np.array(initial_code, dtype=np.int32)
    
    # Track which words are selected
    selected_mask = np.zeros(TOTAL, dtype=bool)
    selected_mask[initial_indices] = True
    selected_count = len(initial_indices)
    
    # For each candidate word, we need to know if it conflicts with any selected word
    # A word conflicts if Hamming distance < 6
    
    # Precompute for each word, the set of words within distance < 6
    # This is expensive to do fully, but we can do it incrementally
    
    # Alternative: maintain a "conflict_count" array
    # conflict_count[w] = number of selected words within distance < 6 from w
    # A word w is addable if conflict_count[w] == 0
    
    # To update conflict_count efficiently, when we select word s, we need to
    # increment conflict_count[w] for all w within distance < 6 from s
    
    # Precompute the list of words within distance < 6 from each word
    # This is still expensive: for each of 131072 words, there are many nearby words
    
    # Let's estimate: number of words within distance 5 of a given word
    # = sum_{k=0}^{5} C(17, k) = 1 + 17 + 136 + 680 + 2380 + 6188 = 9402
    # So for each selected word, we need to update ~9400 entries
    # With ~256+ selected words, that's ~2.5M updates total, which is fine
    
    # Precompute neighbor lists for distance < 6
    # We'll store for each word, the list of words at distance 1-5
    
    # Actually, let's use a different approach:
    # Instead of precomputing all neighbors, we'll compute on-the-fly using numpy
    
    # For efficiency, let's precompute the neighbor lists
    # Memory: 131072 * 9402 * 4 bytes = ~4.9 GB - too much
    
    # Alternative: don't precompute, compute incrementally
    # When selecting word s, find all words within distance < 6 from s
    # This can be done by generating all subsets of positions of size 1-5
    
    # Let's precompute the list of "delta" patterns (bit masks) of weight 1-5
    deltas = []
    for weight in range(1, 6):
        # Generate all bit masks of given weight
        for mask in range(1 << N):
            if mask.bit_count() == weight:
                deltas.append(mask)
    
    deltas = np.array(deltas, dtype=np.uint32)
    num_deltas = len(deltas)
    
    # conflict_count tracks how many selected words are within distance < 6
    conflict_count = np.zeros(TOTAL, dtype=np.int32)
    
    # Initialize conflict_count for the initial code
    for idx in initial_indices:
        word = all_words[idx]
        # For each delta, the neighbor is word XOR delta
        neighbors = word ^ deltas
        # Only count neighbors that are valid (all are, since XOR stays in range)
        # But we need to avoid counting the word itself (delta=0 is not included)
        # Also avoid double-counting if multiple deltas lead to same neighbor (impossible for distinct deltas)
        # Increment conflict_count for these neighbors
        # But we must not count the word itself as conflicting with itself
        # Since deltas have weight >= 1, word ^ delta != word, so no self-conflict
        np.add.at(conflict_count, neighbors, 1)
    
    # Now find all words with conflict_count == 0 (addable)
    # Use MRV heuristic: prioritize words with fewest remaining conflicts
    # Actually, for greedy set packing, we want to add words that have the most
    # "slack" - i.e., words that don't conflict with many potential future additions
    
    # Simpler heuristic: just add all addable words in order of increasing weight
    # Then repeat until no more can be added
    
    # Sort all words by weight
    order = np.lexsort((all_words, weights))
    
    # Greedy loop: keep adding words until no more can be added or time runs out
    while True:
        if time.time() > deadline:
            break
        
        added_any = False
        
        for idx in order:
            if time.time() > deadline:
                break
            
            if selected_mask[idx]:
                continue
            
            if conflict_count[idx] == 0:
                # Add this word
                selected_mask[idx] = True
                selected_count += 1
                added_any = True
                
                word = all_words[idx]
                neighbors = word ^ deltas
                np.add.at(conflict_count, neighbors, 1)
        
        if not added_any:
            break
    
    # Extract the selected words
    result = all_words[selected_mask].tolist()
    
    # Verify minimum distance (sanity check on a sample)
    if len(result) > 1:
        # Check a random sample of pairs
        sample_size = min(100, len(result))
        sample_indices = np.random.choice(len(result), sample_size, replace=False)
        valid = True
        for i in range(sample_size):
            for j in range(i + 1, sample_size):
                wi = result[sample_indices[i]]
                wj = result[sample_indices[j]]
                dist = (wi ^ wj).bit_count()
                if dist < MIN_DIST:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            # Fall back to initial code
            return initial_code
    
    return result
