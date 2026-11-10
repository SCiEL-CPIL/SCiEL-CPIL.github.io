import itertools
import numpy as np
import time

def construct(q, n, radius):
    start_time = time.time()
    limit = 115.0 # Safety margin
    
    total_words = q ** n
    # Map word to index: sum(w[i] * q^(n-1-i))
    
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors for all words
    # neighbors[idx] = list of indices of words within Hamming distance radius
    # For radius 1, this is 1 + n*(q-1)
    neighbors = [None] * total_words
    for idx in range(total_words):
        w = idx_to_word(idx)
        nbrs = [idx]
        for i in range(n):
            orig = w[i]
            for new_val in range(q):
                if new_val == orig:
                    continue
                w[i] = new_val
                nbrs.append(word_to_idx(w))
            w[i] = orig
        neighbors[idx] = nbrs

    uncovered = np.ones(total_words, dtype=bool)
    code = []
    
    # Greedy Set Cover
    while np.any(uncovered):
        if time.time() - start_time > limit:
            break
            
        best_idx = -1
        best_count = -1
        
        # Vectorized count for all candidates?
        # We can create a matrix of shape (total_words, max_neighbors) but sizes vary or we can just loop.
        # Since total_words is small (729), a loop is fine.
        for c in range(total_words):
            # If this candidate is already "used" in a sense? No, we can pick any word in the space.
            # However, picking a word that is already covered doesn't hurt, but might not be optimal if it covers fewer *new* words.
            # The greedy step picks the word that covers the most *uncovered* words.
            
            # Optimization: If the candidate itself is covered, it might still cover many uncovered neighbors.
            # We just count uncovered neighbors.
            
            nbrs = neighbors[c]
            # Count how many of these neighbors are currently uncovered
            # Using numpy sum on a small list might be slower than a loop for very small lists, 
            # but let's try numpy sum on a pre-allocated array if we can, or just sum(uncovered[nbrs])
            # uncovered is a numpy array, nbrs is a list of ints.
            # sum(uncovered[nbrs]) creates a boolean array of length len(nbrs) and sums it.
            
            count = np.sum(uncovered[nbrs])
            
            if count > best_count:
                best_count = count
                best_idx = c
        
        if best_count == 0:
            # Should not happen if the space is connected and radius >= 1
            break
            
        code.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        
    # Convert indices back to words
    result = [idx_to_word(idx) for idx in code]
    
    # Ensure distinctness (greedy ensures we pick new "coverage", but could theoretically pick same word if it covers 0 new? No, we break if 0. 
    # Could we pick the same word twice? 
    # If we pick word X, we mark its neighbors covered. 
    # In the next step, if X is the best, it means it covers some *other* uncovered words? 
    # No, if X was picked, all its neighbors are covered. So next time, count for X will be 0. 
    # So we will never pick the same word twice unless there are multiple components or radius issues, which isn't the case here.)
    
    return result
