"""Improved ternary covering code K_3(6,1) via vectorized greedy + local search."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np


def _all_words(q, n):
    """Generate all q^n words as an (N, n) int array."""
    if n == 0:
        return np.zeros((1, 0), dtype=np.int8)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    return words


def _hamming_within_radius(words, codewords, radius):
    """
    For each word in 'words', check if it's within Hamming distance <= radius of any codeword.
    Returns a boolean array of same length as words.
    Uses vectorized comparison in chunks to avoid memory blowup.
    """
    n = words.shape[1]
    covered = np.zeros(len(words), dtype=bool)
    chunk_size = max(1, 2000000 // max(1, len(codewords)))
    for start in range(0, len(words), chunk_size):
        end = min(start + chunk_size, len(words))
        chunk = words[start:end]
        # shape: (chunk_len, num_codewords, n)
        diff = (chunk[:, None, :] != codewords[None, :, :]).sum(axis=2)
        covered[start:end] = (diff <= radius).any(axis=1)
    return covered


def _greedy_cover(q, n, radius, deadline, initial_codewords=None):
    """
    Greedy set-cover: repeatedly pick the word that covers the most uncovered words.
    Returns list of codewords (list of lists of ints).
    """
    all_words = _all_words(q, n)
    N = len(all_words)
    
    if initial_codewords is None:
        initial_codewords = []
    
    covered = np.zeros(N, dtype=bool)
    codewords = list(initial_codewords)
    
    # Mark initially covered
    if codewords:
        cw = np.array(codewords, dtype=np.int8)
        covered = _hamming_within_radius(all_words, cw, radius)
    
    uncovered_count = N - covered.sum()
    
    # Precompute for each position and value, which words differ only at that position
    # Actually, for greedy, we need to find for each candidate codeword, how many new words it covers.
    # A word w is covered by codeword c if Hamming(w, c) <= radius.
    # For radius=1, w is covered by c if w == c or w differs from c in exactly 1 position.
    
    # For efficiency, precompute for each position i, group words by their value at position i.
    # A codeword c covers: c itself, and all words that differ from c in exactly one position.
    # For radius 1: neighbors of c are words that agree with c in all positions except one.
    
    # Let's precompute: for each position i and value v, the set of indices of words with words[:,i]==v.
    # Then for codeword c, the words it covers are:
    #   - c itself (1 word)
    #   - for each position i: words that differ from c only at position i
    #     = words where words[:,j]==c[j] for all j!=i, and words[:,i]!=c[i]
    
    # This is complex to vectorize easily for arbitrary radius. Let's use a different approach.
    # For small N (729), we can afford to check all candidates each iteration.
    
    start_time = time.time()
    
    while uncovered_count > 0 and time.time() < deadline:
        # Find the best candidate
        # For each word in all_words, compute how many uncovered words it covers
        # A candidate c covers word w if Hamming(c, w) <= radius
        
        # Vectorized: for each candidate c (row in all_words), count uncovered words within radius
        # diff[c_idx, w_idx] = Hamming(all_words[c_idx], all_words[w_idx])
        # This is N x N matrix, 729x729 = 531441 entries, manageable
        
        diff = np.zeros((N, N), dtype=np.int16)
        for i in range(n):
            diff += (all_words[:, i:i+1] != all_words[i, None]).astype(np.int16)
        
        # For each candidate, count how many uncovered words are within radius
        # uncovered_mask: (N,) boolean
        uncovered_mask = ~covered
        # count_new[c] = sum over w where uncovered_mask[w] and diff[c,w] <= radius
        count_new = (diff <= radius).astype(np.int16) @ uncovered_mask.astype(np.int16)
        
        # Pick the candidate with max count_new
        best_idx = np.argmax(count_new)
        if count_new[best_idx] == 0:
            break
        
        new_codeword = all_words[best_idx].tolist()
        codewords.append(new_codeword)
        
        # Update covered
        covered[diff[best_idx] <= radius] = True
        uncovered_count = N - covered.sum()
        
        if time.time() >= deadline:
            break
    
    return codewords


def _local_search(codewords, q, n, radius, deadline, all_words=None):
    """
    Try to remove codewords if possible, or swap to reduce count.
    Simple approach: try removing each codeword and check if coverage is maintained.
    """
    if all_words is None:
        all_words = _all_words(q, n)
    N = len(all_words)
    
    codewords = list(codewords)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        cw = np.array(codewords, dtype=np.int8)
        covered = _hamming_within_radius(all_words, cw, radius)
        
        for i in range(len(codewords)):
            if time.time() >= deadline:
                break
            # Try removing codeword i
            if i == 0:
                remaining = codewords[1:]
            elif i == len(codewords) - 1:
                remaining = codewords[:-1]
            else:
                remaining = codewords[:i] + codewords[i+1:]
            
            if not remaining:
                continue
            
            cw_rem = np.array(remaining, dtype=np.int8)
            covered_rem = _hamming_within_radius(all_words, cw_rem, radius)
            if covered_rem.all():
                codewords = remaining
                improved = True
                break
    
    return codewords


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    all_words = _all_words(q, n)
    
    # Start with a good initial set: the direct sum construction
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        initial = []
    
    # Greedy cover from initial
    codewords = _greedy_cover(q, n, radius, deadline, initial_codewords=initial)
    
    # Local search to remove redundant codewords
    if time.time() < deadline:
        codewords = _local_search(codewords, q, n, radius, deadline, all_words=all_words)
    
    # Verify coverage
    cw = np.array(codewords, dtype=np.int8)
    covered = _hamming_within_radius(all_words, cw, radius)
    if not covered.all():
        # Fallback to full greedy without initial
        codewords = _greedy_cover(q, n, radius, deadline, initial_codewords=[])
    
    return codewords
# EVOLVE-BLOCK-END
