"""
Covering code construction for K_q(n, radius).
Uses a greedy set-cover heuristic with local improvement, starting from a direct-sum construction.
"""
import time
import numpy as np
from itertools import product


def construct(q, n, radius):
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin before 120s limit

    # Generate all words in the space
    total_words = q ** n
    if total_words > 10**7:
        # Too large for full-space verification; fall back to direct sum
        return _direct_sum(q, n, radius)

    # Build the full space as an array of indices
    all_words = np.array(list(product(range(q), repeat=n)), dtype=np.int32)
    
    # For each word, precompute the set of words within Hamming distance <= radius
    # But that's expensive for large spaces. Instead, we'll use a different approach.
    
    # Strategy: Greedy set cover
    # 1. Generate candidate codewords (all words in the space)
    # 2. Iteratively pick the codeword that covers the most uncovered points
    # 3. Use local search to improve
    
    # For efficiency, represent coverage using bitsets or arrays
    # Each point is covered by points within distance <= radius
    
    # Precompute for each candidate codeword, which points it covers
    # This is a matrix of size total_words x total_words, which is too large for big spaces
    
    # Alternative: Use the fact that we only need to cover all points.
    # Start with a good initial solution (direct sum), then try to remove codewords.
    
    # Initial solution: direct sum of tetracode
    initial_code = _direct_sum(q, n, radius)
    
    # Verify coverage and build uncovered set
    covered = _compute_coverage(initial_code, q, n, radius, all_words)
    
    if not np.all(covered):
        # Need to add more codewords to cover uncovered points
        uncovered_indices = np.where(~covered)[0]
        
        while len(uncovered_indices) > 0 and (time.time() - start_time) < DEADLINE:
            # Find the best codeword to add (covers most uncovered points)
            best_idx = -1
            best_count = 0
            
            # Sample a subset of candidates for speed
            num_candidates = min(total_words, 5000)
            candidate_indices = np.random.choice(total_words, size=num_candidates, replace=False)
            
            for ci in candidate_indices:
                word = all_words[ci]
                # Count how many uncovered points are within distance <= radius of word
                dists = _hamming_distances(word, all_words[uncovered_indices], n)
                count = np.sum(dists <= radius)
                if count > best_count:
                    best_count = count
                    best_idx = ci
            
            if best_idx >= 0:
                initial_code.append(all_words[best_idx].tolist())
                covered = _compute_coverage(initial_code, q, n, radius, all_words)
                uncovered_indices = np.where(~covered)[0]
            else:
                break
    
    # Now try to remove redundant codewords
    code = [tuple(w) for w in initial_code]
    
    # Local improvement: try removing each codeword and see if coverage is still maintained
    improved = True
    while improved and (time.time() - start_time) < DEADLINE:
        improved = False
        for i in range(len(code) - 1, -1, -1):
            if (time.time() - start_time) > DEADLINE:
                break
            # Try removing code[i]
            remaining = code[:i] + code[i+1:]
            if _verify_coverage(remaining, q, n, radius, all_words):
                code = remaining
                improved = True
    
    # Try adding better codewords to replace pairs
    for _ in range(50):
        if (time.time() - start_time) > DEADLINE:
            break
        # Randomly pick two codewords, see if one can be replaced by a new one
        if len(code) < 2:
            break
        i, j = np.random.choice(len(code), size=2, replace=False)
        # Try replacing code[i] with a random word
        new_word = tuple(np.random.randint(0, q, size=n))
        if new_word in code:
            continue
        test_code = code[:i] + [new_word] + code[i+1:]
        if len(test_code) < len(code) or _verify_coverage(test_code, q, n, radius, all_words):
            code = test_code
    
    return [list(w) for w in code]


def _direct_sum(q, n, radius):
    """Direct sum of ternary tetracode with all words on remaining coordinates."""
    if n <= 4:
        # For small n, use all words
        return [list(w) for w in product(range(q), repeat=n)]
    
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = []
    for h in tetracode:
        for tail in product(range(q), repeat=n - 4):
            code.append(h + list(tail))
    return code


def _hamming_distances(word, words, n):
    """Compute Hamming distances from word to each word in words."""
    # word: shape (n,), words: shape (m, n)
    return np.sum(words != word, axis=1)


def _compute_coverage(code, q, n, radius, all_words):
    """Return a boolean array indicating which points in all_words are covered."""
    covered = np.zeros(len(all_words), dtype=bool)
    for word in code:
        dists = _hamming_distances(np.array(word, dtype=np.int32), all_words, n)
        covered |= (dists <= radius)
    return covered


def _verify_coverage(code, q, n, radius, all_words):
    """Check if all points are covered by the given code."""
    if len(code) == 0:
        return len(all_words) == 0
    covered = np.zeros(len(all_words), dtype=bool)
    for word in code:
        dists = _hamming_distances(np.array(word, dtype=np.int32), all_words, n)
        covered |= (dists <= radius)
        if np.all(covered):
            return True
    return False
