import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin before 120s limit
    
    # Total number of words
    total_words = q ** n
    
    # Precompute balls for each word
    # A ball is a set of words within Hamming distance 'radius' from the center
    # We represent sets as bitsets (Python integers)
    
    # Helper to convert word (list of ints) to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
        
    # Helper to convert index to word
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word

    # Precompute the ball for each word
    # ball[i] is an integer bitset where bit j is 1 if word j is in the ball of word i
    balls = [0] * total_words
    
    # To compute balls efficiently, we can iterate through all words and their neighbors
    # For each word w, its ball includes w and all words at distance 1..radius
    # Given small n and q, we can just generate neighbors explicitly
    
    # Generate all words as indices
    all_indices = range(total_words)
    
    # Precompute neighbors for distance 1
    # neighbors[i] is a list of indices at distance 1 from i
    neighbors = [[] for _ in range(total_words)]
    
    # This might be slow if done naively for 6561 words.
    # Let's optimize ball generation.
    # Since radius is 1, ball(i) = {i} U {j : dist(i, j) == 1}
    
    # We can compute the bitset for each word by setting its own bit and the bits of its neighbors.
    
    # To find neighbors of i:
    # Convert i to word, change one coordinate to a different value, convert back to index.
    
    for i in range(total_words):
        bitset = 1 << i
        w = idx_to_word(i)
        for pos in range(n):
            orig_val = w[pos]
            for new_val in range(q):
                if new_val != orig_val:
                    w[pos] = new_val
                    j = word_to_idx(w)
                    bitset |= (1 << j)
            w[pos] = orig_val
        balls[i] = bitset

    # Greedy Set Cover
    uncovered = (1 << total_words) - 1  # All bits set
    code = []
    
    # Optimization: Instead of checking all words every time, 
    # we can just iterate. 6561 words * ~500 steps * bit operations is fast.
    
    while uncovered != 0:
        if time.time() > deadline:
            break
            
        best_word = -1
        best_count = -1
        
        # Iterate through all possible words to find the one covering the most uncovered words
        # We can optimize by only considering words that are themselves uncovered or cover many uncovered words
        # But simple iteration is likely fast enough for 6561 words.
        
        for i in range(total_words):
            # Count how many uncovered words are in the ball of i
            # popcount of (balls[i] & uncovered)
            count = (balls[i] & uncovered).bit_count()
            
            if count > best_count:
                best_count = count
                best_word = i
                
                # Early exit if we found a word that covers all remaining uncovered words
                if count == uncovered.bit_count():
                    break
        
        if best_word == -1 or best_count == 0:
            # This should not happen if uncovered is not 0, unless we ran out of time or logic error
            # If best_count is 0, it means no word covers any uncovered word, which is impossible
            # unless the space is disconnected, which it isn't.
            break
            
        code.append(idx_to_word(best_word))
        uncovered &= ~balls[best_word]
        
        # Optional: Random tie-breaking to explore different local optima?
        # For now, deterministic greedy is a good baseline.
        
    return code
