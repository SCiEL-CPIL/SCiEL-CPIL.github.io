import time
import numpy as np

def construct():
    """
    Construct the largest set of 17-bit words with pairwise Hamming distance >= 6.
    Uses a greedy algorithm with conflict tracking and a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    N = 17
    MIN_DIST = 6
    TOTAL_WORDS = 1 << N  # 131072
    
    # Precompute bit masks for positions
    # We'll work with integer representations
    
    # Strategy: 
    # 1. Start with a known good code (the 256-word linear code from the seed's extension)
    # 2. Try to add more words greedily using a conflict matrix
    # 3. Use vectorized operations for speed
    
    # First, let's construct the 256-word linear code
    # The [18,9,6] extended quadratic-residue code shortened by one coordinate
    # For now, let's use a simpler approach: start with the 32-word Reed-Muller code
    # and try to expand it
    
    # Actually, let's use a different approach:
    # Generate all 2^17 words, then use greedy set packing
    
    # For efficiency, we'll use a numpy-based approach
    # But 2^17 = 131072 words, and we need to compute pairwise distances
    # That's 131072^2 / 2 pairs, which is too many
    
    # Better approach: use a greedy algorithm that processes words in order
    # and maintains a list of selected words with their conflict status
    
    # Let's use a different strategy:
    # 1. Start with the known 256-word code
    # 2. Try to add additional words one by one
    # 3. Use vectorized distance computation for speed
    
    # Construct the 256-word linear code
    # The [18,9,6] code shortened: we can generate it using the quadratic residue code
    # For simplicity, let's use a systematic construction
    
    # Actually, let's use a simpler approach: 
    # Start with the 32-word first-order Reed-Muller code on 16 coordinates
    # and try to extend it to 17 coordinates with distance 6
    
    words = []
    
    # Generate the 32-word RM(1,4) code on 16 coordinates
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    
    # Now we have 32 words on 16 coordinates. We need to extend to 17 coordinates.
    # Let's try adding a 17th bit (position 16) to each word, trying both 0 and 1
    # and checking if the distance constraint is still satisfied
    
    # Actually, let's use a completely different approach:
    # Use a greedy algorithm that processes all 2^17 words in a smart order
    
    # Precompute the bit patterns for all words
    # This is memory intensive but should fit in memory
    # 131072 words * 8 bytes each = ~1MB, which is fine
    
    # Let's use a different strategy: 
    # 1. Generate candidate words in a specific order
    # 2. For each candidate, check if it's at distance >= 6 from all selected words
    # 3. If yes, add it to the selected set
    
    # To make this efficient, we'll use numpy for vectorized distance computation
    
    # First, let's create an array of all possible words
    all_words = np.arange(1 << N, dtype=np.uint32)
    
    # We'll process words in a specific order to maximize the final code size
    # A good heuristic is to process words with lower weight first (or some other order)
    
    # For now, let's just process in numerical order and see how far we get
    
    selected = []
    selected_np = np.array([], dtype=np.uint32)
    
    # We'll use a conflict count array to track how many conflicts each word has
    # with the selected set
    
    # Actually, let's use a simpler approach:
    # Maintain the selected words in a numpy array
    # For each candidate, compute distances to all selected words using vectorized ops
    
    # But this is O(n^2) in the number of selected words, which could be slow
    
    # Let's use a different strategy: 
    # Use a bit-parallel approach to compute conflicts
    
    # For each selected word, precompute a mask of words that conflict with it
    # Then for each candidate, check if it conflicts with any selected word
    
    # This is also O(n^2) but with better constants
    
    # Actually, let's just use a straightforward greedy approach with early termination
    # and a deadline guard
    
    # Precompute the bit masks for each position
    # This will help with fast distance computation
    
    # Let's use a different approach: 
    # Use the fact that we only need distance >= 6
    # We can use a lookup table or vectorized operations
    
    # For now, let's use a simple greedy approach with numpy
    
    # Convert all_words to a 2D array of bits for fast distance computation
    # Shape: (131072, 17)
    
    # This is 131072 * 17 * 1 byte = ~2.2MB, which is fine
    
    # Extract bits
    bits = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        bits[:, i] = (all_words >> i) & 1
    
    # Now we can compute Hamming distances using XOR and sum
    
    # Greedy algorithm:
    # 1. Start with an empty set
    # 2. For each word (in some order), check if it's at distance >= 6 from all selected words
    # 3. If yes, add it to the selected set
    
    # To make this efficient, we'll process words in a specific order
    # Let's try processing in order of increasing weight (number of 1s)
    
    # Compute weights
    weights = bits.sum(axis=1)
    
    # Sort by weight, then by value
    order = np.lexsort((all_words, weights))
    
    # Greedy selection
    selected_indices = []
    
    for idx in order:
        if time.time() > deadline:
            break
        
        word = all_words[idx]
        
        # Check if this word is at distance >= 6 from all selected words
        if not selected_indices:
            selected_indices.append(idx)
            continue
        
        # Compute distances to all selected words
        selected_bits = bits[selected_indices]
        diff = bits[idx] ^ selected_bits
        distances = diff.sum(axis=1)
        
        if np.all(distances >= MIN_DIST):
            selected_indices.append(idx)
    
    # Convert selected indices to actual words
    result = all_words[selected_indices].tolist()
    
    # Ensure we have at least the 32-word base code
    # If the greedy approach didn't find enough, fall back to the base code
    if len(result) < 32:
        words = []
        for slope in range(16):
            for intercept in range(2):
                word = 0
                for point in range(16):
                    bit = ((slope & point).bit_count() + intercept) % 2
                    word |= bit << point
                words.append(word)
        return words
    
    return result
