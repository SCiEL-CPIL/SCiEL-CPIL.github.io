"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np


def _hamming_dist(a, b):
    return sum(1 for i in range(len(a)) if a[i] != b[i])


def _covers(codeword, q, n, radius):
    # yield all words within Hamming distance <= radius of codeword
    # efficient for small radius
    if radius == 0:
        yield tuple(codeword)
        return
    if radius == 1:
        yield tuple(codeword)
        for i in range(n):
            for v in range(q):
                if v != codeword[i]:
                    w = list(codeword)
                    w[i] = v
                    yield tuple(w)
        return
    # general case: enumerate all words within distance <= radius
    # For radius=1 this is O(n*q), for larger it's more complex
    # Use combinations of positions to change
    for r in range(radius + 1):
        if r == 0:
            yield tuple(codeword)
        else:
            for positions in itertools.combinations(range(n), r):
                for changes in itertools.product(range(q), repeat=r):
                    w = list(codeword)
                    for pos, val in zip(positions, changes):
                        if val != w[pos]:
                            w[pos] = val
                        else:
                            # skip if no actual change (but we need exactly r positions changed)
                            # Actually for covering, we want distance <= radius, so we can have fewer changes
                            # Let's just include all
                            pass
                    yield tuple(w)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    total_words = q ** n
    
    # Start with the tetracode construction as initial solution
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    else:
        # Fallback: greedy construction
        initial = []
    
    # Greedy set cover approach
    # Each codeword covers (n+1)*(q-1) + 1 words for radius 1
    # We need to cover all q^n words
    
    if n >= 4 and q == 3:
        codewords = list(initial)
    else:
        codewords = []
    
    # Track which words are covered using a boolean array
    # For q=3, n=7, total = 2187, manageable
    covered = [False] * total_words
    
    def word_to_idx(word):
        idx = 0
        for v in word:
            idx = idx * q + v
        return idx
    
    def idx_to_word(idx):
        word = []
        for _ in range(n):
            word.append(idx % q)
            idx //= q
        return word
    
    # Mark initially covered words
    for cw in codewords:
        for w in _covers(cw, q, n, radius):
            idx = word_to_idx(list(w))
            covered[idx] = True
    
    uncovered_count = total_words - sum(covered)
    
    # Greedy: add codewords that cover the most uncovered words
    while uncovered_count > 0:
        if time.time() > deadline - 5:
            break
        
        best_cw = None
        best_gain = 0
        best_gain_words = []
        
        # Try all possible codewords
        for cw_tuple in itertools.product(range(q), repeat=n):
            gain = 0
            gain_words = []
            for w in _covers(list(cw_tuple), q, n, radius):
                idx = word_to_idx(list(w))
                if not covered[idx]:
                    gain += 1
                    gain_words.append(idx)
            if gain > best_gain:
                best_gain = gain
                best_cw = list(cw_tuple)
                best_gain_words = gain_words
        
        if best_gain == 0:
            break
        
        codewords.append(best_cw)
        for idx in best_gain_words:
            covered[idx] = True
        uncovered_count -= best_gain
    
    # Local search: try to remove codewords
    if time.time() < deadline - 3:
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for i in range(len(codewords)):
                # Temporarily remove codeword i
                removed = codewords.pop(i)
                # Check if all words still covered
                still_covered = True
                for j in range(len(codewords)):
                    if j == i:
                        continue
                    for w in _covers(codewords[j], q, n, radius):
                        idx = word_to_idx(list(w))
                        if not covered[idx]:
                            # This word was only covered by removed
                            # Actually we need to check if this word is covered by remaining
                            pass
                
                # Better approach: recompute coverage
                temp_covered = [False] * total_words
                for j in range(len(codewords)):
                    for w in _covers(codewords[j], q, n, radius):
                        idx = word_to_idx(list(w))
                        temp_covered[idx] = True
                
                if all(temp_covered):
                    covered = temp_covered
                    changed = True
                else:
                    codewords.insert(i, removed)
    
    # Ensure validity
    final_covered = [False] * total_words
    for cw in codewords:
        for w in _covers(cw, q, n, radius):
            idx = word_to_idx(list(w))
            final_covered[idx] = True
    
    if not all(final_covered):
        # Fallback to tetracode construction
        if n >= 4 and q == 3:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            codewords = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return codewords
# EVOLVE-BLOCK-END
