"""Greedy set-cover for ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools


def _all_words(q, n):
    """Generate all q^n words as a 2D numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.uint8)
    for i in range(total):
        v = i
        for j in range(n):
            words[i, j] = v % q
            v //= q
    return words


def _neighbors_indices(word, q, n):
    """Return indices of all words within Hamming distance 1 of `word` (including itself)."""
    # word is a 1D array of length n
    neighbors = [word.copy()]
    for pos in range(n):
        original = word[pos]
        for val in range(q):
            if val == original:
                continue
            new_word = word.copy()
            new_word[pos] = val
            neighbors.append(new_word)
    return neighbors


def construct(q, n, radius):
    # For the specific task: q=3, n=6, radius=1
    # Total space size: 3^6 = 729
    total = q ** n
    all_words = _all_words(q, n)
    
    # Precompute for each word, the set of indices it covers (within radius)
    # We'll use a bitset or boolean mask approach
    # Since total is small (729), we can use boolean arrays
    
    # Precompute coverage sets for all possible codewords
    # coverage[i] = boolean array of length total, True if word i is within radius of codeword i
    # Actually, we need: for each candidate codeword c, which words does it cover?
    # Let's precompute: cover_mask[c] = set of word indices within Hamming distance <= radius of c
    
    # For radius=1, each codeword covers 1 + n*(q-1) = 1 + 6*2 = 13 words
    # Let's precompute the coverage for each of the 729 possible codewords
    
    # Build a mapping from word tuple to index
    word_to_idx = {}
    for i in range(total):
        word_to_idx[tuple(all_words[i])] = i
    
    # Precompute coverage for each candidate
    # coverage[c] is a set of indices
    coverage = [set() for _ in range(total)]
    for c in range(total):
        cw = all_words[c]
        # Add self
        coverage[c].add(c)
        # Add all words at Hamming distance 1
        for pos in range(n):
            orig = cw[pos]
            for val in range(q):
                if val == orig:
                    continue
                new_word = cw.copy()
                new_word[pos] = val
                idx = word_to_idx.get(tuple(new_word))
                if idx is not None:
                    coverage[c].add(idx)
    
    # Greedy set cover
    uncovered = set(range(total))
    codewords = []
    
    while uncovered:
        best_c = -1
        best_count = -1
        for c in range(total):
            count = len(coverage[c] & uncovered)
            if count > best_count:
                best_count = count
                best_c = c
                if best_count == len(uncovered):
                    break
        if best_c == -1:
            break
        codewords.append(all_words[best_c].tolist())
        uncovered -= coverage[best_c]
    
    # Remove duplicates (shouldn't happen, but just in case)
    seen = set()
    unique_codewords = []
    for cw in codewords:
        t = tuple(cw)
        if t not in seen:
            seen.add(t)
            unique_codewords.append(cw)
    
    return unique_codewords
# EVOLVE-BLOCK-END
