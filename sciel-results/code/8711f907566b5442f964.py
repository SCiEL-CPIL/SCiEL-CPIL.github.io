"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np
import random

def _ball_indices(word, q, n, radius):
    """Return the flat indices of all words within Hamming distance `radius` of `word`."""
    idxs = []
    if radius == 1:
        idxs.append(int(np.ravel_multi_index(tuple(word), (q,)*n)))
        for pos in range(n):
            for v in range(q):
                if v == word[pos]:
                    continue
                w2 = list(word)
                w2[pos] = v
                idxs.append(int(np.ravel_multi_index(tuple(w2), (q,)*n)))
    return idxs

def _tetracode(q, n, radius):
    """Direct product: tetracode on first 4 coords, all words on remaining n-4 coords."""
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

def _construct_k3_6_1(q, n, radius, deadline):
    """
    Specialized construction for q=3, n=6, radius=1.
    Uses numpy-vectorized greedy set cover followed by local search (remove/swap).
    """
    total = q ** n
    all_words = list(itertools.product(range(q), repeat=n))
    
    # Precompute coverage matrix: cov[i, j] = 1 if word j is within radius of word i
    cov = np.zeros((total, total), dtype=np.uint8)
    for i, w in enumerate(all_words):
        idxs = _ball_indices(list(w), q, n, radius)
        for idx in idxs:
            cov[i, idx] = 1
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=np.uint8)
    code_indices = []
    
    while uncovered.any():
        # Find center that covers most uncovered points
        # cov[:, uncovered] gives coverage of uncovered points
        coverage_counts = cov[:, uncovered].sum(axis=1)
        best_idx = np.argmax(coverage_counts)
        
        if coverage_counts[best_idx] == 0:
            break
            
        code_indices.append(best_idx)
        uncovered[cov[best_idx] == 1] = 0
        
        if time.time() > deadline - 5:
            break
    
    # Local search: try to remove redundant codewords
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(code_indices)):
            # Check if removing code_indices[i] still covers everything
            # Temporarily remove
            temp_uncovered = np.zeros(total, dtype=np.uint8)
            for j, idx in enumerate(code_indices):
                if j != i:
                    temp_uncovered[cov[idx] == 1] = 1
            # The points covered by code_indices[i] that are NOT covered by others
            # are the ones that would become uncovered
            would_lose = (cov[code_indices[i]] == 1) & (temp_uncovered == 0)
            
            if not would_lose.any():
                # Safe to remove
                code_indices.pop(i)
                improved = True
                break
    
    # Local search: try swaps
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(len(code_indices)):
            # Try replacing code_indices[i] with some other center
            # First, find what would be lost if we remove code_indices[i]
            temp_uncovered = np.zeros(total, dtype=np.uint8)
            for j, idx in enumerate(code_indices):
                if j != i:
                    temp_uncovered[cov[idx] == 1] = 1
            would_lose = (cov[code_indices[i]] == 1) & (temp_uncovered == 0)
            
            if not would_lose.any():
                continue  # Already handled by removal
                
            # Try to find a center that covers all of would_lose and adds new coverage
            best_replace = -1
            best_gain = 0
            for c in range(total):
                if c in code_indices:
                    continue
                # Check if c covers all of would_lose
                if not (cov[c, would_lose] == 1).all():
                    continue
                # Calculate gain: new points covered by c that weren't covered before
                # Points covered by c but not by current code (excluding i)
                new_covered = (cov[c] == 1) & (temp_uncovered == 0)
                gain = new_covered.sum()
                if gain > best_gain:
                    best_gain = gain
                    best_replace = c
            
            if best_replace >= 0 and best_gain > 0:
                code_indices[i] = best_replace
                improved = True
                break
    
    # Convert to codewords
    result = [list(all_words[i]) for i in code_indices]
    
    # Verify coverage
    final_uncovered = np.ones(total, dtype=np.uint8)
    for idx in code_indices:
        final_uncovered[cov[idx] == 1] = 0
    
    if not final_uncovered.any():
        return result
    else:
        # Fallback to tetracode if local search failed
        return _tetracode(q, n, radius)

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    if q == 3 and n == 6 and radius == 1:
        return _construct_k3_6_1(q, n, radius, deadline)
    else:
        return _tetracode(q, n, radius)
# EVOLVE-BLOCK-END
