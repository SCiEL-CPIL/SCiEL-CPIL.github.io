"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import numpy as np

# Pre-compute the ternary tetracode (9 words of length 4, perfect covering radius 1)
_TETRACODE = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
_TETRACODE_SET = frozenset(tuple(t) for t in _TETRACODE)


def _is_valid_code(codewords, q, n, radius):
    """Check if every word in {0..q-1}^n is within Hamming distance radius of some codeword."""
    # Build a set of all covered words
    covered = set()
    for cw in codewords:
        # Generate all words within Hamming distance radius
        for delta in range(radius + 1):
            if delta == 0:
                covered.add(tuple(cw))
            else:
                for positions in itertools.combinations(range(n), delta):
                    for values in itertools.product(range(q), repeat=delta):
                        new_word = list(cw)
                        for pos, val in zip(positions, values):
                            new_word[pos] = val
                        covered.add(tuple(new_word))
    # Check if all words are covered
    total_words = q ** n
    return len(covered) == total_words


def _try_remove_redundant(codewords, q, n, radius, time_limit):
    """Try to remove redundant codewords from a valid code."""
    codewords = list(codewords)
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for i in range(len(codewords)):
            if time.time() - start_time > time_limit:
                return codewords
            # Try removing codeword i
            candidate = codewords[:i] + codewords[i + 1:]
            if _is_valid_code(candidate, q, n, radius):
                codewords = candidate
                improved = True
                break
    return codewords


def _local_search(codewords, q, n, radius, time_limit):
    """Perform local search by swapping codewords to improve coverage balance."""
    codewords = list(codewords)
    start_time = time.time()
    rng = random.Random(42)
    
    # Compute coverage counts
    all_words = list(itertools.product(range(q), repeat=n))
    
    while time.time() - start_time < time_limit:
        # Compute coverage count for each word
        coverage = [0] * len(all_words)
        for cw in codewords:
            for word in all_words:
                dist = sum(1 for a, b in zip(cw, word) if a != b)
                if dist <= radius:
                    coverage[all_words.index(word)] += 1
        
        # Find words with minimum coverage
        min_cov = min(coverage)
        if min_cov > 1:
            # Try to find a codeword that can be replaced
            # For each codeword, check if removing it and adding a new one maintains validity
            for i in range(len(codewords)):
                if time.time() - start_time > time_limit:
                    return codewords
                # Try adding a random new codeword
                new_cw = [rng.randrange(q) for _ in range(n)]
                if new_cw in codewords:
                    continue
                candidate = codewords[:i] + [new_cw] + codewords[i + 1:]
                if _is_valid_code(candidate, q, n, radius):
                    codewords = candidate
                    break
    return codewords


def construct(q, n, radius):
    # Direct sum of the ternary tetracode with every word on the other n-4 coordinates
    if n <= 4:
        # For small n, use a simple approach
        if n == 4:
            return [list(t) for t in _TETRACODE]
        elif n < 4:
            # Generate all words and filter
            all_words = list(itertools.product(range(q), repeat=n))
            # Greedy set cover
            codewords = []
            uncovered = set(all_words)
            while uncovered:
                best_word = None
                best_count = -1
                for w in uncovered:
                    count = sum(1 for u in uncovered if sum(1 for a, b in zip(w, u) if a != b) <= radius)
                    if count > best_count:
                        best_count = count
                        best_word = w
                codewords.append(list(best_word))
                uncovered = set(u for u in uncovered if sum(1 for a, b in zip(best_word, u) if a != b) > radius)
            return codewords
    
    # Start with the tetracode construction
    codewords = [h + list(tail) for h in _TETRACODE for tail in itertools.product(range(q), repeat=n - 4)]
    
    # Try to remove redundant codewords
    codewords = _try_remove_redundant(codewords, q, n, radius, time_limit=60)
    
    return codewords
# EVOLVE-BLOCK-END
