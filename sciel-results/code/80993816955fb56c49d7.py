"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

import numpy as np


def _tetracode():
    return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]


def _neighbors(word, q, n):
    """Yield all words within Hamming distance 1 of word."""
    yield tuple(word)
    for i in range(n):
        for v in range(q):
            if v != word[i]:
                w = list(word)
                w[i] = v
                yield tuple(w)


def _covers(word, q, n, uncovered):
    """Return the set of uncovered positions that word would cover."""
    covered = set()
    for nb in _neighbors(word, q, n):
        if nb in uncovered:
            covered.add(nb)
    return covered


def _build_greedy(q, n, radius, deadline, seed=42):
    """Greedy set-cover style construction."""
    rng = random.Random(seed)
    total = q ** n
    # Represent all words as integers 0..q^n - 1
    # A word w is covered by codeword c if Hamming distance <= radius
    # We'll track uncovered words as a set of integers
    
    # Precompute: for each possible codeword, which words it covers
    # This is too large to precompute for n=7, q=3 (2187 words, each covers 1+7*2=15 neighbors)
    # So we compute on the fly.
    
    def word_to_int(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def int_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Start with all words uncovered
    uncovered = set(range(total))
    codewords = []
    
    # Precompute neighbors for each integer word
    # neighbor_map[codeword_int] = set of word_ints it covers
    # For radius=1: covers itself + all words differing in exactly 1 position
    neighbor_cache = {}
    
    def get_neighbors(ci):
        if ci in neighbor_cache:
            return neighbor_cache[ci]
        word = int_to_word(ci)
        nbs = {ci}
        for pos in range(n):
            for v in range(q):
                if v != word[pos]:
                    new_word = list(word)
                    new_word[pos] = v
                    nbs.add(word_to_int(new_word))
        neighbor_cache[ci] = nbs
        return nbs
    
    # Greedy: repeatedly pick the codeword that covers the most uncovered words
    # To speed up, we can sample candidate codewords from the uncovered set
    while uncovered and time.time() < deadline:
        # Sample some candidates from uncovered
        candidates = list(uncovered)
        if len(candidates) > 200:
            candidates = rng.sample(candidates, 200)
        
        best_ci = None
        best_count = -1
        for ci in candidates:
            nbs = get_neighbors(ci)
            count = sum(1 for nb in nbs if nb in uncovered)
            if count > best_count:
                best_count = count
                best_ci = ci
        
        if best_ci is None or best_count <= 0:
            break
        
        nbs = get_neighbors(best_ci)
        uncovered -= nbs
        codewords.append(int_to_word(best_ci))
    
    return codewords


def _try_remove(codewords, q, n, radius, uncovered_set):
    """Try to remove a codeword if it's redundant. Returns (new_codewords, new_uncovered, removed_idx or None)."""
    # For each codeword, check if all words it uniquely covers are covered by others
    # This is expensive; we do a quick check by removing and seeing if coverage breaks
    # We'll track for each word which codewords cover it
    
    # Build: for each word, list of codeword indices that cover it
    # For radius=1, each codeword covers 15 words
    # Total entries: len(codewords) * 15
    
    def word_to_int(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def int_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # coverage[word_int] = set of codeword indices
    coverage = defaultdict(set)
    for ci, cw in enumerate(codewords):
        word = cw
        nbs = {word_to_int(word)}
        for pos in range(n):
            for v in range(q):
                if v != word[pos]:
                    new_word = list(word)
                    new_word[pos] = v
                    nbs.add(word_to_int(new_word))
        for nb in nbs:
            coverage[nb].add(ci)
    
    # Try to remove codewords that are not the sole cover of any word
    removable = []
    for ci in range(len(codewords)):
        word = codewords[ci]
        nbs = {word_to_int(word)}
        for pos in range(n):
            for v in range(q):
                if v != word[pos]:
                    new_word = list(word)
                    new_word[pos] = v
                    nbs.add(word_to_int(new_word))
        is_redundant = True
        for nb in nbs:
            if len(coverage[nb]) == 1:
                is_redundant = False
                break
        if is_redundant:
            removable.append(ci)
    
    # Remove all redundant codewords
    if not removable:
        return codewords, None, None
    
    new_codewords = [cw for i, cw in enumerate(codewords) if i not in set(removable)]
    return new_codewords, None, removable


def _local_search(codewords, q, n, radius, deadline, seed=0):
    """Local search: try swaps and removals to reduce code size."""
    rng = random.Random(seed)
    total = q ** n
    
    def word_to_int(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def int_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute neighbors for all possible codewords (2187 of them, each 15 neighbors)
    all_neighbors = {}
    for ci in range(total):
        word = int_to_word(ci)
        nbs = {ci}
        for pos in range(n):
            for v in range(q):
                if v != word[pos]:
                    new_word = list(word)
                    new_word[pos] = v
                    nbs.add(word_to_int(new_word))
        all_neighbors[ci] = nbs
    
    # Current coverage: for each word, how many codewords cover it
    cover_count = [0] * total
    for cw in codewords:
        ci = word_to_int(cw)
        for nb in all_neighbors[ci]:
            cover_count[nb] += 1
    
    # Verify full coverage
    assert all(c > 0 for c in cover_count), "Coverage broken!"
    
    best_code = list(codewords)
    best_size = len(codewords)
    
    # Try removals first
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_code)):
            if time.time() >= deadline:
                break
            # Try removing codeword i
            ci = word_to_int(best_code[i])
            nbs = all_neighbors[ci]
            can_remove = all(cover_count[nb] > 1 for nb in nbs)
            if can_remove:
                # Remove it
                for nb in nbs:
                    cover_count[nb] -= 1
                best_code.pop(i)
                best_size -= 1
                improved = True
                break
    
    # Try swaps: replace one codeword with another
    for _ in range(2000):
        if time.time() >= deadline:
            break
        if len(best_code) <= best_size:
            continue
        
        # Pick a random codeword to try replacing
        i = rng.randint(0, len(best_code) - 1)
        old_ci = word_to_int(best_code[i])
        old_nbs = all_neighbors[old_ci]
        
        # Words only covered by this codeword
        only_covered = [nb for nb in old_nbs if cover_count[nb] == 1]
        
        if not only_covered:
            # Can just remove
            for nb in old_nbs:
                cover_count[nb] -= 1
            best_code.pop(i)
            best_size -= 1
            continue
        
        # Find a new codeword that covers all only_covered words
        # Sample candidates
        found = False
        for _ in range(50):
            new_ci = rng.randint(0, total - 1)
            if new_ci == old_ci:
                continue
            new_nbs = all_neighbors[new_ci]
            if all(nb in new_nbs for nb in only_covered):
                # Check if this swap reduces size or maintains it
                # After swap, words in old_nbs but not in new_nbs lose coverage
                lost = [nb for nb in old_nbs if nb not in new_nbs]
                if lost:
                    # Need those to still be covered
                    can_swap = all(cover_count[nb] > 1 for nb in lost)
                    if not can_swap:
                        continue
                
                # Perform swap
                for nb in old_nbs:
                    cover_count[nb] -= 1
                for nb in new_nbs:
                    cover_count[nb] += 1
                best_code[i] = int_to_word(new_ci)
                found = True
                break
        
        # Also try adding a new codeword then removing two old ones (size -1)
        # This is more complex; skip for now
    
    return best_code


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Phase 1: Greedy construction
    code = _build_greedy(q, n, radius, deadline - 60, seed=42)
    
    if time.time() > deadline - 50:
        return code
    
    # Phase 2: Local search with removals and swaps
    code = _local_search(code, q, n, radius, deadline - 20, seed=0)
    
    # Phase 3: Try a few more restarts if time permits
    if time.time() < deadline - 30:
        for seed in range(1, 5):
            if time.time() > deadline - 30:
                break
            code2 = _build_greedy(q, n, radius, deadline - 30, seed=seed)
            code2 = _local_search(code2, q, n, radius, deadline - 10, seed=seed)
            if len(code2) < len(code):
                code = code2
    
    return code
# EVOLVE-BLOCK-END
