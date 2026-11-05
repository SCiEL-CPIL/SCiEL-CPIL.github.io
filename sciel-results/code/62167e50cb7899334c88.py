"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    if q != 3 or n != 8 or radius != 1:
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words

    # Build initial code: tetracode prefix + all tails
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    m = len(code)
    
    # Map word to index
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx

    # Precompute neighborhoods as list of lists of indices
    neighborhoods = []
    for c in code:
        neigh = [word_to_idx(c)]
        for pos in range(n):
            orig = c[pos]
            for val in range(3):
                if val != orig:
                    new_c = c[:]
                    new_c[pos] = val
                    neigh.append(word_to_idx(new_c))
        neighborhoods.append(neigh)

    # Coverage count
    coverage = np.zeros(6561, dtype=np.int32)
    for neigh in neighborhoods:
        for idx in neigh:
            coverage[idx] += 1

    # Track active codewords
    active = list(range(m))
    
    # Peeling: iteratively remove the most redundant removable codeword
    while True:
        best_idx = -1
        best_min_cov = 0  # We want to maximize the minimum coverage in neighborhood
        
        for i in active:
            # Check removability and compute min coverage in neighborhood
            min_cov = 255
            for idx in neighborhoods[i]:
                cov = coverage[idx]
                if cov < min_cov:
                    min_cov = cov
            if min_cov >= 2:
                # Removable; prefer higher min_cov (more redundant)
                if min_cov > best_min_cov:
                    best_min_cov = min_cov
                    best_idx = i
        
        if best_idx < 0:
            break
        
        # Remove best_idx
        for idx in neighborhoods[best_idx]:
            coverage[idx] -= 1
        active.remove(best_idx)
    
    return [code[i] for i in active]
# EVOLVE-BLOCK-END
