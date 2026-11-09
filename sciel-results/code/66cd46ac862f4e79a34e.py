"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

import numpy as np

# Precompute the ternary tetracode (perfect code of radius 1 over length 4)
_TETRACODE = np.array(
    [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
)


def _tetracode_cover_set():
    """Return a numpy boolean array of shape (3**4,) marking words covered by the tetracode."""
    covered = np.zeros(3 ** 4, dtype=bool)
    for h in _TETRACODE:
        idx = h[0] * 27 + h[1] * 9 + h[2] * 3 + h[3]
        covered[idx] = True
        for pos in range(4):
            for val in range(3):
                if val == h[pos]:
                    continue
                new_h = h.copy()
                new_h[pos] = val
                idx2 = new_h[0] * 27 + new_h[1] * 9 + new_h[2] * 3 + new_h[3]
                covered[idx2] = True
    return covered


# Precompute which of the 81 tail words are "covered" by the tetracode when combined with each tetracode word.
# Actually, for the direct sum construction:
# A codeword is (h, t) where h is a tetracode word (len 4) and t is any word of length n-4.
# The coverage: a word (u, v) is covered if there exists h in tetracode, t in tails, and a position
# such that changing at most one coordinate makes (u,v) == (h,t).
# This is equivalent to: the distance from (u,v) to the set {(h,t)} is <= 1.
# This happens iff:
#   dist(u, tetracode) <= 1 AND dist(v, {t}) = 0  (change in first 4 coords or no change)
#   OR dist(u, tetracode) = 0 AND dist(v, {t}) <= 1  (change in last n-4 coords)
# Wait, that's not quite right. The distance is the sum of distances in each part if we allow at most 1 total change.
# So: (u,v) is covered iff min_{h in tetracode, t in tails} (dist(u,h) + dist(v,t)) <= 1.
# This means either:
#   1. dist(u,h) = 0 and dist(v,t) <= 1 for some h,t
#   2. dist(u,h) <= 1 and dist(v,t) = 0 for some h,t
# So the covered set is: { (u,v) : (dist(u,T)<=1 and v in tails) OR (u in T and dist(v,tails)<=1) }
# But tails is the entire space {0,1,2}^{n-4}, so dist(v,tails)=0 always, and dist(v,tails)<=1 always.
# So the covered set is actually the ENTIRE space. That's the point of the direct sum construction.
# The number of codewords is 9 * 3**(n-4).

# Let's think of a better approach. We want to reduce the number of codewords.
# The direct sum gives 9 * 3**(n-4) codewords. For n=6, that's 9*9=81.
# The best known is 73. So we need to do better.

# Strategy: Start with the direct sum construction (81 codewords for n=6) and try to remove some codewords
# while maintaining coverage. We can use a greedy approach: try to remove each codeword and check if coverage is maintained.

def _compute_coverage(codewords, q, n, radius):
    """Compute which words are covered by the given codewords. Returns a boolean array of size q**n."""
    num_words = q ** n
    covered = np.zeros(num_words, dtype=bool)
    
    # Convert codewords to indices
    cw_indices = []
    for cw in codewords:
        idx = 0
        for i, val in enumerate(cw):
            idx = idx * q + val
        cw_indices.append(idx)
    
    for idx in cw_indices:
        # Mark the word itself and all words at distance <= radius
        # For radius 1, this is the word itself plus all words differing in exactly one position
        covered[idx] = True
        # For radius 1, we need to check all neighbors
        word = [0] * n
        temp = idx
        for i in range(n - 1, -1, -1):
            word[i] = temp % q
            temp //= q
        
        for pos in range(n):
            for val in range(q):
                if val == word[pos]:
                    continue
                new_word = word.copy()
                new_word[pos] = val
                new_idx = 0
                for i, v in enumerate(new_word):
                    new_idx = new_idx * q + v
                covered[new_idx] = True
    
    return covered


def _is_covered(codewords, q, n, radius):
    """Check if all words are covered."""
    covered = _compute_coverage(codewords, q, n, radius)
    return np.all(covered)


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145  # 5 second safety margin
    
    # Start with the direct sum construction
    if n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tails = list(itertools.product(range(q), repeat=n - 4))
        codewords = [h + list(tail) for h in tetracode for tail in tails]
    else:
        # For n < 4, use brute force
        codewords = [list(word) for word in itertools.product(range(q), repeat=n)]
    
    # Try to remove codewords greedily
    # Order codewords by some heuristic to try removal in a good order
    # We'll try to remove codewords that have the least "unique" coverage
    
    # First, compute coverage for each codeword
    num_codewords = len(codewords)
    
    # Compute coverage count for each word in the space
    # A word is "uniquely covered" by a codeword if that codeword is the only one covering it
    # We want to remove codewords that don't have any uniquely covered words
    
    # For efficiency, let's use a different approach:
    # Try removing each codeword one at a time, in random order, and check if coverage is maintained
    
    best_codewords = codewords[:]
    best_len = len(codewords)
    
    # Randomize the order of codewords to try removal
    indices = list(range(num_codewords))
    random.shuffle(indices)
    
    # Try to remove codewords in random order
    for i in indices:
        if time.time() > deadline:
            break
        
        # Try removing codeword i
        new_codewords = [cw for j, cw in enumerate(codewords) if j != i]
        
        # Check if coverage is maintained
        if _is_covered(new_codewords, q, n, radius):
            codewords = new_codewords
            best_codewords = codewords[:]
            best_len = len(codewords)
            # Re-index
            indices = list(range(len(codewords)))
            random.shuffle(indices)
    
    return best_codewords
# EVOLVE-BLOCK-END
