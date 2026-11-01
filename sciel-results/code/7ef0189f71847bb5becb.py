"""Greedy set-cover for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools


def construct(q, n, radius):
    """
    Construct a covering code using a greedy set-cover heuristic.
    Optimized for q=3, n=6, radius=1 using NumPy vectorization.
    """
    total = q ** n
    
    # Generate all words in lexicographical order (base-q)
    # This ensures index consistency with base-q expansion
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Precompute the coverage matrix C
    # C[i, j] = 1 if word j is within Hamming distance 'radius' of word i
    # This is the "ball" of word i.
    C = np.zeros((total, total), dtype=np.uint8)
    
    # Precompute powers of q for index calculation
    # index = sum(w[k] * q^(n-1-k))
    # We can calculate the weight of each position
    weights = [q ** (n - 1 - k) for k in range(n)]
    
    for i in range(total):
        w = all_words[i]
        C[i, i] = 1
        
        if radius >= 1:
            for pos in range(n):
                orig_val = w[pos]
                weight = weights[pos]
                for new_val in range(q):
                    if new_val == orig_val:
                        continue
                    new_idx = i - orig_val * weight + new_val * weight
                    C[i, new_idx] = 1
    
    # Greedy Set Cover
    # uncovered[j] is True if word j is not yet covered
    uncovered = np.ones(total, dtype=bool)
    codewords = []
    
    while np.any(uncovered):
        # Calculate how many uncovered words each candidate covers
        # scores[i] = sum(uncovered[j] for j where C[i, j] == 1)
        # Using dot product for vectorization
        scores = C.dot(uncovered.astype(np.uint8))
        
        # Select the candidate covering the most uncovered words
        best_c = np.argmax(scores)
        
        # If no candidate covers any new words, we are stuck (should not happen)
        if scores[best_c] == 0:
            break
            
        # Add the selected codeword to the result
        codewords.append(all_words[best_c].tolist())
        
        # Mark all words covered by best_c as covered
        uncovered[C[best_c] == 1] = False
        
    return codewords
# EVOLVE-BLOCK-END
