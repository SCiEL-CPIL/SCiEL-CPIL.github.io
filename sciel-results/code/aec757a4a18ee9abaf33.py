# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct(q, n, radius):
    """
    Construct a covering code using Greedy Set Cover followed by Greedy Removal.
    
    1. Precompute neighbors for all q^n words.
    2. Greedily select codewords that cover the most currently uncovered words.
    3. Post-process by trying to remove any codeword that is redundant.
    """
    total = q ** n
    
    # Helper to encode/decode
    def encode(word):
        v = 0
        for sym in word:
            v = v * q + sym
        return v

    def decode(v):
        out = [0] * n
        for i in range(n - 1, -1, -1):
            v, r = divmod(v, q)
            out[i] = r
        return out

    # Precompute neighbors for each word
    # neighbors[w] is a list of indices of words within Hamming distance <= radius
    neighbors = [None] * total
    for w in range(total):
        word = decode(w)
        neigh = set([w])
        
        # Generate all words within distance radius
        # For small n and q, we can just iterate all words and check distance, 
        # but for efficiency we generate by flipping positions.
        # Since q=3, n=6, radius=1, the neighbor size is small (13).
        # For general radius, we can use combinations.
        
        # Efficient generation for small radius:
        # We iterate over all subsets of positions of size <= radius.
        # For each subset, we change the symbols to something different.
        # This is O(C(n, r) * (q-1)^r).
        
        # Simpler: Just iterate all 729 words and check Hamming distance.
        # 729 * 729 = ~500k operations. Very fast.
        for u in range(total):
            if u == w:
                continue
            u_word = decode(u)
            dist = sum(1 for i in range(n) if word[i] != u_word[i])
            if dist <= radius:
                neigh.add(u)
        
        neighbors[w] = list(neigh)

    # 1. Greedy Set Cover
    uncovered = set(range(total))
    codewords = []
    
    while uncovered:
        best_cw = None
        best_count = 0
        
        # Find the word that covers the most uncovered words
        # We can optimize by only checking candidates that are in uncovered or 
        # whose neighbors intersect uncovered. But checking all 729 is fine.
        for cw in range(total):
            # If this codeword doesn't cover any new uncovered words, skip
            # count = len(set(neighbors[cw]) & uncovered)
            count = 0
            for nb in neighbors[cw]:
                if nb in uncovered:
                    count += 1
                    if count > best_count:
                        # Early exit not possible easily without sorting, 
                        # but we can break if count == len(uncovered)
                        pass
            
            if count > best_count:
                best_count = count
                best_cw = cw
        
        if best_cw is None:
            break # Should not happen if graph is connected and radius >= 1
            
        codewords.append(best_cw)
        # Remove covered words from uncovered
        for nb in neighbors[best_cw]:
            if nb in uncovered:
                uncovered.remove(nb)

    # 2. Greedy Removal (Post-processing)
    # Try to remove each codeword if it doesn't uncover any words.
    # We maintain a coverage count array.
    coverage_count = [0] * total
    for cw in codewords:
        for nb in neighbors[cw]:
            coverage_count[nb] += 1
            
    i = 0
    while i < len(codewords):
        cw = codewords[i]
        # Temporarily remove
        for nb in neighbors[cw]:
            coverage_count[nb] -= 1
            
        # Check if any word has coverage 0
        if all(c > 0 for c in coverage_count):
            # Keep removed
            codewords.pop(i)
        else:
            # Restore
            for nb in neighbors[cw]:
                coverage_count[nb] += 1
            i += 1

    # Convert to lists of integers
    result = [decode(cw) for cw in codewords]
    return result
# EVOLVE-BLOCK-END
