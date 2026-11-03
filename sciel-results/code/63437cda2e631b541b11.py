"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Generate all words in {0,...,q-1}^n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total = all_words.shape[0]  # q**n
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    # For radius=1, neighbors are the word itself and all words differing in exactly 1 position.
    # Number of neighbors = 1 + n * (q - 1)
    neighbors = []
    for i in range(total):
        w = all_words[i]
        neigh = [i]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w2 = w.copy()
                w2[pos] = val
                # Calculate index of w2
                idx = 0
                for j in range(n):
                    idx = idx * q + int(w2[j])
                neigh.append(idx)
        neighbors.append(neigh)
    
    # Initialize coverage counts
    # cover_count[v] = number of codewords in the current set that cover v
    cover_count = np.zeros(total, dtype=np.int32)
    
    # Initially, all words are in the set.
    # So, for each word c, it covers all its neighbors.
    for c in range(total):
        for v in neighbors[c]:
            cover_count[v] += 1
            
    # We will remove words if they are redundant.
    # A word c is redundant if for all v in neighbors[c], cover_count[v] > 1.
    # This means some other word in the set also covers v.
    
    # We process words in a random order to avoid bias.
    order = list(range(total))
    random.shuffle(order)
    
    active = np.ones(total, dtype=bool)
    
    for c in order:
        if not active[c]:
            continue
            
        # Check if c is redundant
        # Get neighbors of c
        neigh = neighbors[c]
        
        # Check min coverage among neighbors
        # If min cover_count > 1, then removing c won't leave any neighbor uncovered by others
        if np.min(cover_count[neigh]) > 1:
            # Remove c
            active[c] = False
            # Update cover counts
            cover_count[neigh] -= 1
            
    # Collect remaining active words
    selected_indices = np.where(active)[0]
    codewords = [all_words[i].tolist() for i in selected_indices]
    
    return codewords
# EVOLVE-BLOCK-END
