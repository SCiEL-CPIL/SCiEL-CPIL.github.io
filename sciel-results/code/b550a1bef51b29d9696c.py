"""Rewritten program: randomized greedy set-cover + deterministic peeling for K_q(n, radius)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

import numpy as np

_DEADLINE_MARGIN = 10.0  # seconds of safety margin before the 150 s kill
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _remaining():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN


def _word_to_int(word, q, n):
    """Convert a word (list/tuple of ints 0..q-1) to a single integer in base q."""
    v = 0
    for i in range(n):
        v = v * q + word[i]
    return v


def _int_to_word(v, q, n):
    """Convert an integer back to a word."""
    w = [0] * n
    for i in range(n - 1, -1, -1):
        w[i] = v % q
        v //= q
    return w


def _neighbors(word, q, n, radius=1):
    """Yield all words within Hamming distance <= radius from word.
    For radius=1: the word itself plus all single-position changes."""
    # Yield the word itself
    yield tuple(word)
    for pos in range(n):
        orig = word[pos]
        for val in range(q):
            if val == orig:
                continue
            new_word = list(word)
            new_word[pos] = val
            yield tuple(new_word)


def _greedy_cover(q, n, radius, seed, time_limit):
    """Randomized greedy set-cover: repeatedly pick the codeword covering the most uncovered points."""
    rng = random.Random(seed)
    total = q ** n
    # We'll track uncovered points as a set of integers
    uncovered = set(range(total))
    code = []

    # Precompute for each possible codeword position (int) the set of points it covers
    # But that's too large. Instead, for each candidate codeword, compute coverage on the fly.
    # Better: iterate over all possible words, compute how many uncovered points each covers,
    # pick the best, add it, update uncovered.
    
    # For efficiency with q=3, n=8: total = 6561, each codeword covers 1 + n*(q-1) = 1+16 = 17 points
    # So we can afford to check all 6561 candidates each iteration.
    
    all_words = [list(range(0, q, 1)) for _ in range(0)]  # placeholder
    # Actually generate all words as tuples
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    
    # Map each word to its covered points (as a frozenset of ints)
    # Precompute coverage for all words
    coverage = []
    for w in all_words:
        pts = set()
        for nb in _neighbors(w, q, n, radius):
            pts.add(_word_to_int(nb, q, n))
        coverage.append(pts)
    
    # Greedy: each step, find the word with max |coverage[word] ∩ uncovered|
    # To speed up, we can sample or use a heuristic.
    # For 6561 words, checking all each time is 6561 * |uncovered| which is too slow initially.
    # Instead, we can use a lazy approach: maintain for each word a count of how many of its points are uncovered.
    
    # Let's use a different approach: maintain an array cover_count[i] = number of uncovered points covered by word i.
    # Initially, cover_count[i] = len(coverage[i]) for all i.
    # When we pick word j, for each point p in coverage[j], we need to decrement cover_count[i] for all i where p in coverage[i].
    # This requires knowing which words cover each point.
    
    # Build point_to_words: for each point, which words cover it.
    point_to_words = defaultdict(list)
    for i, cov in enumerate(coverage):
        for p in cov:
            point_to_words[p].append(i)
    
    cover_count = [len(cov) for cov in coverage]
    uncovered_set = set(range(total))
    
    while uncovered_set:
        if _remaining() < 5.0:
            break
        # Find the word with max cover_count
        # Use argmax
        best_idx = max(range(len(all_words)), key=lambda i: cover_count[i])
        if cover_count[best_idx] == 0:
            break
        
        # Add this word to code
        code.append(all_words[best_idx])
        
        # Update: for each point in coverage[best_idx] that is still uncovered,
        # remove it from uncovered_set and decrement cover_count for all words covering that point
        for p in coverage[best_idx]:
            if p in uncovered_set:
                uncovered_set.remove(p)
                for i in point_to_words[p]:
                    cover_count[i] -= 1
    
    return [list(w) for w in code]


def _peeling_reduce(code, q, n, radius, time_limit):
    """Deterministic peeling: iteratively remove codewords whose removal doesn't break coverage."""
    code = [list(w) for w in code]
    if not code:
        return code
    
    total = q ** n
    # Build coverage for each codeword
    def get_coverage(w):
        pts = set()
        for nb in _neighbors(w, q, n, radius):
            pts.add(_word_to_int(nb, q, n))
        return pts
    
    # Precompute coverage for all codewords
    coverages = [get_coverage(w) for w in code]
    
    # For each point, count how many codewords cover it
    point_count = defaultdict(int)
    for cov in coverages:
        for p in cov:
            point_count[p] += 1
    
    changed = True
    while changed and _remaining() > 5.0:
        changed = False
        for i in range(len(code)):
            if _remaining() < 5.0:
                break
            # Check if removing codeword i leaves all points still covered
            removable = True
            for p in coverages[i]:
                if point_count[p] <= 1:
                    removable = False
                    break
            if removable:
                # Remove codeword i
                for p in coverages[i]:
                    point_count[p] -= 1
                code.pop(i)
                coverages.pop(i)
                changed = True
                break  # Restart the loop since indices changed
    
    return code


def _swap_improve(code, q, n, radius, time_limit):
    """Try to improve by swapping one codeword for another that covers more uncovered-when-removed points.
    Actually, try removing one and seeing if we can add a better one, or just try to find a smaller set.
    For now, a simple local search: try replacing each codeword with a random one and see if coverage is maintained."""
    code = [list(w) for w in code]
    if not code:
        return code
    
    total = q ** n
    
    def get_coverage(w):
        pts = set()
        for nb in _neighbors(w, q, n, radius):
            pts.add(_word_to_int(nb, q, n))
        return pts
    
    # Current coverage
    all_covered = set()
    for w in code:
        all_covered |= get_coverage(w)
    
    # Point counts
    point_count = defaultdict(int)
    for w in code:
        cov = get_coverage(w)
        for p in cov:
            point_count[p] += 1
    
    rng = random.Random(42)
    
    improved = True
    while improved and _remaining() > 10.0:
        improved = False
        for i in range(len(code)):
            if _remaining() < 10.0:
                break
            # Try to find a replacement for code[i] that allows removal
            # Actually, try removing code[i] and see if we can cover the newly uncovered points with one new word
            old_cov = get_coverage(code[i])
            # Points that would become uncovered if we remove code[i]
            newly_uncovered = set()
            for p in old_cov:
                if point_count[p] <= 1:
                    newly_uncovered.add(p)
            
            if not newly_uncovered:
                # Can simply remove
                for p in old_cov:
                    point_count[p] -= 1
                code.pop(i)
                improved = True
                break
            
            # Try to find a single word that covers all newly_uncovered points
            # This is a hitting set problem. For small newly_uncovered, we can search.
            if len(newly_uncovered) <= 20:
                # Find candidate words that cover at least one newly uncovered point
                # and check if any covers all
                best_word = None
                best_overlap = 0
                # Sample some candidates
                for _ in range(200):
                    cand = [rng.randint(0, q - 1) for _ in range(n)]
                    cov = get_coverage(cand)
                    overlap = len(cov & newly_uncovered)
                    if overlap == len(newly_uncovered):
                        best_word = cand
                        break
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_word = cand
                
                if best_word is not None and best_overlap == len(newly_uncovered):
                    # Replace code[i] with best_word
                    # But we also need to make sure best_word doesn't introduce issues
                    # Actually, we're replacing, so we remove old and add new
                    for p in old_cov:
                        point_count[p] -= 1
                    new_cov = get_coverage(best_word)
                    for p in new_cov:
                        point_count[p] += 1
                    code[i] = best_word
                    improved = True
                    break
    
    return code


def construct(q, n, radius):
    global _START
    _START = time.time()
    
    # Use greedy set-cover
    code = _greedy_cover(q, n, radius, seed=42, time_limit=None)
    
    # Peel to reduce
    code = _peeling_reduce(code, q, n, radius, time_limit=None)
    
    # Try swap improvements
    if _remaining() > 15.0:
        code = _swap_improve(code, q, n, radius, time_limit=None)
    
    # Final peeling
    code = _peeling_reduce(code, q, n, radius, time_limit=None)
    
    # Ensure distinctness
    seen = set()
    unique_code = []
    for w in code:
        t = tuple(w)
        if t not in seen:
            seen.add(t)
            unique_code.append(w)
    
    return unique_code
# EVOLVE-BLOCK-END
