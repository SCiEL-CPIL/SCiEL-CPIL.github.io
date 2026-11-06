import time
import numpy as np
from itertools import combinations

N = 18
MIN_DISTANCE = 6
TOTAL = 1 << N
DEADLINE_MARGIN = 5.0

def _seed_words():
    """First-order Reed-Muller code on 16 coordinates, extended by two zero bits.
    32 words, pairwise distance 8 (>= 6)."""
    words = []
    for slope in range(16):
        for intercept in range(2):
            word = 0
            for point in range(16):
                bit = ((slope & point).bit_count() + intercept) % 2
                word |= bit << point
            words.append(word)
    return words

def _get_neighbors(center, radius):
    """Generate all words within Hamming distance <= radius of center."""
    results = [center]
    bits = list(range(N))
    for r in range(1, radius + 1):
        for comb in combinations(bits, r):
            w = center
            for b in comb:
                w ^= (1 << b)
            results.append(w)
    return results

def _build_forbidden_mask(words):
    """Build a boolean mask of size 2^N where True means 'forbidden'."""
    forbidden = np.zeros(TOTAL, dtype=bool)
    for w in words:
        neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
        forbidden[neighbors] = True
    return forbidden

def _generate_words_by_weight():
    """
    Generator that yields integers in [0, 2^N) sorted by Hamming weight (popcount),
    then by value. This systematic traversal ensures we explore dense regions
    of the Hamming space first, which often contain more valid codewords for
    constant weight or near-constant weight codes.
    """
    # We can't store all 2^18 words in a list and sort them easily in memory/time 
    # if we do it naively, but 2^18 is 262,144. 
    # Creating a list of 262k ints and sorting by bit_count is actually quite fast 
    # in Python (usually < 0.1s). Let's do that to allow random access or just iteration.
    
    # Actually, generating all and sorting is O(N * 2^N). 
    # 262k * 18 operations ~ 4.7M ops. Very fast.
    
    all_words = np.arange(TOTAL, dtype=np.int64)
    # np.bitwise_count is not standard in older numpy, but Python 3.10+ has int.bit_count.
    # For numpy array, we can use:
    # weights = np.array([x.bit_count() for x in all_words]) -> slow in Python loop.
    
    # Better: Use lookup table for byte weights or just rely on the fact that 
    # we can iterate. But sorting is key for the strategy.
    
    # Let's use a vectorized way to get weights if possible, or just a fast loop.
    # Actually, we can just generate the list and sort with key=lambda x: x.bit_count()
    # This creates a list of 262k tuples (weight, value). 
    # Sorting 262k items is fast.
    
    # To be safe and fast, let's precompute the sorted list once.
    # Since this is inside the function, we should cache it or compute it efficiently.
    # Given the 60s limit, computing it once at start is fine.
    
    # Optimization: We can use the fact that weights are small (0..18).
    # We can create 19 lists, one for each weight, and fill them.
    # This is O(2^N) to fill, O(2^N) to iterate.
    
    buckets = [[] for _ in range(N + 1)]
    for w in range(TOTAL):
        buckets[w.bit_count()].append(w)
        
    # Now yield from buckets in order of weight
    for weight in range(N + 1):
        for w in buckets[weight]:
            yield w

def _systematic_greedy(deadline):
    words = _seed_words()
    forbidden = _build_forbidden_mask(words)
    
    start_time = time.time()
    
    # Pre-generate the sorted list of all words by weight
    # This takes ~0.1-0.2s for 2^18
    all_words_by_weight = list(_generate_words_by_weight())
    
    # We will iterate through this list.
    # To speed up, we can check the forbidden mask in bulk?
    # No, we need to add words one by one because adding a word changes the forbidden mask.
    # However, we can skip large chunks if they are already forbidden.
    
    # But the forbidden mask changes.
    # So we must check each candidate.
    
    # Optimization: 
    # Instead of checking every single word in the sorted list (262k checks),
    # we can just iterate. 262k checks is very fast (< 1s).
    # The bottleneck is _get_neighbors and mask updating.
    # _get_neighbors(5) generates 12616 neighbors.
    # If we add 100 words, that's 1.2M mask updates.
    # Total words we might add? A_2(18,6) is around 1000-2000?
    # If we add 1000 words, 12M mask updates. 
    # 12M boolean assignments in NumPy is fast (~0.1s).
    # So the total time should be well under 60s.
    
    added_count = 0
    for w in all_words_by_weight:
        if time.time() > deadline:
            break
            
        if not forbidden[w]:
            # Candidate is valid
            words.append(w)
            added_count += 1
            
            # Update forbidden mask
            neighbors = _get_neighbors(w, MIN_DISTANCE - 1)
            forbidden[neighbors] = True
            
    return words

def construct():
    start = time.time()
    deadline = start + 90.0 - DEADLINE_MARGIN

    best_words = None
    best_len = 0

    # We only run the systematic search once, as it is deterministic and exhaustive 
    # (within the constraints of the greedy choice). 
    # Random restarts might help if the greedy choice is bad, but systematic 
    # weight-ordering is a strong heuristic for code construction.
    
    try:
        w = _systematic_greedy(deadline)
        if len(w) > best_len:
            best_len = len(w)
            best_words = w
    except Exception:
        pass

    if best_words is None:
        best_words = _seed_words()

    # Deduplicate and validate
    seen = set()
    out = []
    for w in best_words:
        if w not in seen and 0 <= w < TOTAL:
            seen.add(w)
            out.append(w)
    if not out:
        out = _seed_words()

    return out
