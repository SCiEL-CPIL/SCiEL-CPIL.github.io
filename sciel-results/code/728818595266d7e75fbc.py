"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
import random

_DEADLINE = 140.0  # safety margin before 150s kill

def _all_words_array(q, n):
    """Return all q^n words as a numpy array of shape (q^n, n)."""
    total = q ** n
    words = np.zeros((total, n), dtype=np.int8)
    for i in range(n):
        words[:, i] = (np.arange(total) // (q ** (n - 1 - i))) % q
    return words

def _hamming_ball_indices(words, center, radius, q):
    """Return boolean mask of words within Hamming distance <= radius from center."""
    # words: (N, n) array, center: (n,) array
    dists = np.sum(words != center, axis=1)
    return dists <= radius

def _greedy_cover(q, n, radius, all_words, time_limit):
    """Greedy set-cover: repeatedly pick the word covering most uncovered words."""
    N = len(all_words)
    uncovered = np.ones(N, dtype=bool)
    codewords = []
    start_time = time.time()
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        # Find the best center: the word that covers the most uncovered words
        # Sample candidate centers from uncovered words for efficiency
        uncov_indices = np.where(uncovered)[0]
        
        # If few uncovered, just pick each
        if len(uncov_indices) <= 50:
            for idx in uncov_indices:
                center = all_words[idx]
                mask = _hamming_ball_indices(all_words, center, radius, q)
                uncovered[mask] = False
                codewords.append(center.tolist())
            break
        
        # Sample candidates from uncovered words
        n_samples = min(len(uncov_indices), 200)
        sample_idx = np.random.choice(uncov_indices, size=n_samples, replace=False)
        sample_centers = all_words[sample_idx]
        
        # For each candidate center, count how many uncovered words it covers
        best_count = -1
        best_center = None
        
        # Vectorized: compute distances from each sample center to all uncovered words
        # This is expensive, so we do it in chunks
        uncov_words = all_words[uncovered]
        
        for ci in range(n_samples):
            center = sample_centers[ci]
            mask = _hamming_ball_indices(uncov_words, center, radius, q)
            count = mask.sum()
            if count > best_count:
                best_count = count
                best_center = center
        
        if best_center is None:
            break
            
        # Add this center to codewords and update uncovered
        full_mask = _hamming_ball_indices(all_words, best_center, radius, q)
        uncovered[full_mask] = False
        codewords.append(best_center.tolist())
    
    return codewords

def _verify_coverage(codewords, q, n, radius):
    """Check that every word is within radius of some codeword."""
    all_words = _all_words_array(q, n)
    N = len(all_words)
    covered = np.zeros(N, dtype=bool)
    cw = np.array(codewords, dtype=np.int8)
    
    for i in range(len(cw)):
        dists = np.sum(all_words != cw[i], axis=1)
        covered[dists <= radius] = True
    
    return covered.all()

def construct(q, n, radius):
    start_time = time.time()
    remaining = _DEADLINE - (time.time() - start_time)
    
    if remaining < 5:
        # Not enough time for greedy; fall back to direct sum
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Generate all words
    all_words = _all_words_array(q, n)
    
    # Try greedy with a time budget
    best_result = None
    best_len = float('inf')
    
    # Multiple restarts if time allows
    time_per_restart = max(5.0, remaining / 3)
    
    for restart in range(3):
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        random.seed(restart * 12345 + 6789)
        np.random.seed(restart * 12345 + 6789)
        
        result = _greedy_cover(q, n, radius, all_words, time_per_restart)
        
        if result is not None and len(result) < best_len:
            # Verify coverage
            if _verify_coverage(result, q, n, radius):
                best_len = len(result)
                best_result = result
        
        if time.time() - start_time > _DEADLINE - 15:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback: direct sum construction
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
# EVOLVE-BLOCK-END
