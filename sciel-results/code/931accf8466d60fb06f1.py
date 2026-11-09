"""
Ternary covering code K_3(8,1) construction.

Strategy:
1. Start with the direct product of the ternary tetracode (9 words, length 4, perfect
   covering radius 1) with all words on the remaining 4 coordinates: 9 * 3^4 = 729 words.
2. Iteratively remove words that are "redundant" – i.e., every word within Hamming
   distance 1 of the removed word is still within Hamming distance 1 of some other
   remaining codeword.
3. Use a greedy ordering: remove words whose removal is "safe" and which cover the
   fewest uniquely-covered words (i.e., words that have the least unique coverage).

This is a structural change from the seed (which just returns the 729-word product code)
to an active pruning/search approach that can find smaller valid covering codes.
"""

import time
import numpy as np
from itertools import product


def construct(q, n, radius):
    start_time = time.time()
    deadline = 150.0  # safety margin before 120s limit
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for general parameters: brute-force product code
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in product(range(3), repeat=n-4)]
        else:
            # Small n: just enumerate all
            return [list(w) for w in product(range(q), repeat=n)]
    
    # Generate the initial 729-word product code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_words = [h + list(tail) for h in tetracode for tail in product(range(3), repeat=4)]
    
    # Represent words as tuples for fast hashing
    words = set(tuple(w) for w in initial_words)
    
    # Precompute all 3^8 = 6561 possible words
    all_words = set(tuple(w) for w in product(range(3), repeat=8))
    
    # For each word, compute the set of words within Hamming distance <= 1
    # A word covers itself and all words at distance 1 (8 positions * 2 alternatives = 16 neighbors + self = 17)
    
    def neighbors(word):
        """Return the set of words within Hamming distance 1 of word."""
        result = {word}
        for i in range(n):
            for v in range(q):
                if v != word[i]:
                    new_word = list(word)
                    new_word[i] = v
                    result.add(tuple(new_word))
        return result
    
    # Build neighbor sets for all words in the code
    # This is expensive for 729 words * 17 neighbors each, but manageable
    
    # Instead, let's work with a different approach:
    # For each codeword, track which "unique" words it covers (words that only this codeword covers)
    
    # First, compute the union of all covered words
    covered = set()
    for w in words:
        covered.update(neighbors(w))
    
    # For efficiency, let's use a different strategy:
    # Greedy removal: repeatedly try to remove words, checking if coverage is maintained.
    # To make this efficient, we'll track for each word in the space, how many codewords cover it.
    
    # Count coverage: for each word in all_words, how many codewords are within distance 1?
    # This is 6561 * 729 checks initially, which is ~4.8M operations - feasible.
    
    # Actually, let's be smarter. For each codeword, we know its 17 neighbors.
    # We can build a count array.
    
    # Map each word to an index
    word_to_idx = {w: i for i, w in enumerate(sorted(all_words))}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    num_words = len(all_words)
    
    # Coverage count: how many codewords cover each word in the space
    cov_count = np.zeros(num_words, dtype=np.int32)
    
    # For each codeword, increment coverage for its neighbors
    cw_indices = []  # indices of current codewords
    for w in words:
        idx = word_to_idx[w]
        cw_indices.append(idx)
        for nb in neighbors(w):
            cov_count[word_to_idx[nb]] += 1
    
    # Now greedy removal:
    # A codeword can be removed if all words it uniquely covers (cov_count == 1) 
    # are... wait, no. We need: after removal, every word still has cov_count >= 1.
    # So a codeword can be removed if for every word w in its neighbor set, cov_count[w] >= 2.
    
    # Let's track which codewords are active
    active = [True] * len(cw_indices)
    active_count = len(cw_indices)
    
    # For efficiency, precompute neighbor indices for each codeword
    cw_neighbor_indices = []
    for idx in cw_indices:
        w = idx_to_word[idx]
        nbs = [word_to_idx[nb] for nb in neighbors(w)]
        cw_neighbor_indices.append(nbs)
    
    # Greedy: repeatedly find a removable codeword and remove it
    # Priority: remove codewords that have the fewest "unique" neighbors (cov_count == 1)
    
    removable = []
    for i in range(len(cw_indices)):
        if not active[i]:
            continue
        # Check if all neighbors have cov_count >= 2
        unique_count = 0
        for nb_idx in cw_neighbor_indices[i]:
            if cov_count[nb_idx] == 1:
                unique_count += 1
        if unique_count == 0:
            removable.append((0, i))  # priority 0, no unique neighbors
        else:
            removable.append((unique_count, i))
    
    # Sort by unique_count (ascending) to remove most redundant first
    removable.sort()
    
    # Process removals
    changed = True
    while changed:
        changed = False
        removable.sort()
        for priority, i in removable:
            if not active[i]:
                continue
            # Re-check: are all neighbors still covered by >= 2 codewords?
            can_remove = True
            for nb_idx in cw_neighbor_indices[i]:
                if cov_count[nb_idx] <= 1:
                    can_remove = False
                    break
            if can_remove:
                # Remove codeword i
                active[i] = False
                active_count -= 1
                for nb_idx in cw_neighbor_indices[i]:
                    cov_count[nb_idx] -= 1
                changed = True
                # Check time
                if time.time() > deadline - 30:
                    break
        if time.time() > deadline - 30:
            break
    
    # Extract remaining codewords
    result = [list(idx_to_word[idx]) for i, idx in enumerate(cw_indices) if active[i]]
    
    return result
