import time
import random
import numpy as np

def _hamming_dist(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

def _neighbors(word, q, n):
    """Yield all words at Hamming distance 1 from word."""
    for pos in range(n):
        orig = word[pos]
        for v in range(q):
            if v != orig:
                w = list(word)
                w[pos] = v
                yield tuple(w)

def _build_cover_check(codewords, q, n):
    """Build a set of covered words for a given list of codewords."""
    covered = set()
    for cw in codewords:
        cw = tuple(cw)
        covered.add(cw)
        for nb in _neighbors(cw, q, n):
            covered.add(nb)
    return covered

def _is_valid_cover(codewords, q, n, radius=1):
    """Check if codewords form a valid covering code."""
    covered = set()
    for cw in codewords:
        cw = tuple(cw)
        covered.add(cw)
        if radius >= 1:
            for nb in _neighbors(cw, q, n):
                covered.add(nb)
    # Check if all words are covered
    total_words = q ** n
    return len(covered) == total_words

def _greedy_cover(q, n, radius, initial_words=None):
    """Greedy algorithm to build a covering code."""
    if initial_words is None:
        initial_words = []
    
    # Precompute all words
    all_words = [list(w) for w in np.ndindex(*([q] * n))]
    all_words_tuples = [tuple(w) for w in all_words]
    word_to_idx = {w: i for i, w in enumerate(all_words_tuples)}
    
    # For each word, compute which words it covers
    word_cov = [set() for _ in range(len(all_words))]
    for i, w in enumerate(all_words):
        wt = tuple(w)
        word_cov[i].add(i)
        if radius >= 1:
            for nb in _neighbors(wt, q, n):
                if nb in word_to_idx:
                    word_cov[i].add(word_to_idx[nb])
    
    covered = set()
    for cw in initial_words:
        cw = tuple(cw)
        idx = word_to_idx[cw]
        covered.update(word_cov[idx])
    
    remaining = len(all_words) - len(covered)
    max_iter = 1000
    for _ in range(max_iter):
        if remaining <= 0:
            break
        best_idx = -1
        best_new = 0
        for i in range(len(all_words)):
            if i in covered:
                continue
            new = len(word_cov[i] - covered)
            if new > best_new:
                best_new = new
                best_idx = i
        if best_idx < 0:
            break
        initial_words.append(all_words[best_idx])
        covered.update(word_cov[best_idx])
        remaining = len(all_words) - len(covered)
    
    return initial_words

def _local_search(codewords, q, n, radius, time_budget):
    """Local search to reduce the number of codewords while maintaining coverage."""
    deadline = time.time() + time_budget
    codewords = [list(cw) for cw in codewords]
    
    # Precompute neighbors for each codeword
    def get_covered(cw_list):
        covered = set()
        for cw in cw_list:
            cw_t = tuple(cw)
            covered.add(cw_t)
            for nb in _neighbors(cw_t, q, n):
                covered.add(nb)
        return covered
    
    current_covered = get_covered(codewords)
    total_words = q ** n
    
    # Precompute all words
    all_words = [tuple(w) for w in np.ndindex(*([q] * n))]
    
    # For local search, we try to remove redundant codewords
    # A codeword is redundant if its coverage is entirely covered by others
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to remove each codeword
        for i in range(len(codewords)):
            if time.time() > deadline:
                return codewords
            # Check if removing codeword[i] still gives full coverage
            test_covered = get_covered(codewords[:i] + codewords[i+1:])
            if len(test_covered) == total_words:
                codewords.pop(i)
                current_covered = test_covered
                improved = True
                break
    
    # Try swapping: replace one codeword with another that gives same or better coverage
    all_words_list = [list(w) for w in all_words]
    for _ in range(100):  # limited swaps
        if time.time() > deadline:
            break
        if len(codewords) <= 0:
            break
        # Try to find a better codeword to replace one
        best_swap = None
        best_gain = 0
        for i in range(len(codewords)):
            if time.time() > deadline:
                return codewords
            # Current coverage without this codeword
            covered_without = get_covered(codewords[:i] + codewords[i+1:])
            # Try replacing with random words
            for trial in range(20):
                if time.time() > deadline:
                    return codewords
                cand = random.choice(all_words_list)
                if cand in codewords and cand != codewords[i]:
                    continue
                new_covered = get_covered(codewords[:i] + [cand] + codewords[i+1:])
                if len(new_covered) > len(current_covered):
                    gain = len(new_covered) - len(current_covered)
                    if gain > best_gain:
                        best_gain = gain
                        best_swap = (i, cand, new_covered)
        if best_swap:
            i, cand, new_covered = best_swap
            codewords[i] = cand
            current_covered = new_covered
    
    return codewords

def _fallback(q, n, radius):
    """Fallback to the seed approach."""
    import itertools
    if q == 3 and n == 6 and radius == 1:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    # Generic greedy
    return _greedy_cover(q, n, radius)

def construct(q, n, radius):
    start_time = time.time()
    # Use most of the time budget for search
    search_time = min(100, max(1, 110 - (time.time() - start_time)))
    
    if q != 3 or n != 6 or radius != 1:
        return _fallback(q, n, radius)
    
    # Start with a good initial solution
    # Use the tetracode-based construction (81 words) as initial
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Verify initial is valid
    if not _is_valid_cover(initial, q, n, radius):
        return initial
    
    # Local search to reduce size
    result = _local_search(initial, q, n, radius, search_time)
    
    # Final validity check
    if _is_valid_cover(result, q, n, radius):
        return result
    
    # If local search broke validity, return the original
    return initial

import itertools
