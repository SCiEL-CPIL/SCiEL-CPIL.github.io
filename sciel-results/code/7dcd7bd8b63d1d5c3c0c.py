"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import numpy as np


def construct(q, n, radius):
    # Use a peeling heuristic based on the dual distance-2 packing.
    # For q=3, n=8, radius=1:
    # 1. Generate a valid covering code (initially the direct sum construction).
    # 2. Iteratively remove the codeword whose 17-neighborhood is most densely covered
    #    by other selected codewords (maximizing redundancy).
    
    if q != 3 or n != 8 or radius != 1:
        # Fallback for other parameters: use the direct sum construction
        # For general case, we'll use a simple greedy or the tetracode direct sum if applicable
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        else:
            # Brute force for small n
            all_words = list(itertools.product(range(q), repeat=n))
            return all_words
    
    # Specific optimization for q=3, n=8, radius=1
    # Start with the direct sum of the ternary tetracode
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=4)]
    
    # Convert to numpy array for efficient operations
    code_arr = np.array(code, dtype=np.int8)
    m = len(code_arr)
    
    if m == 0:
        return code
    
    # For each codeword, compute its neighborhood (itself + 8 neighbors at distance 1)
    # A codeword c covers: c, and c with one coordinate changed to each of the other 2 values
    # Total 1 + 8*2 = 17 words per codeword
    
    # We'll use a peeling approach:
    # 1. Build a coverage count array for all 3^8 = 6561 words
    # 2. Find the codeword whose neighborhood is most "redundant" (covered most by others)
    # 3. Remove it if coverage remains valid
    # 4. Repeat until no more removals are possible
    
    # Map from word tuple to index in 0..6560
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * 3 + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = idx % 3
            idx //= 3
        return word
    
    # Precompute neighborhoods for each codeword
    # neighborhood[i] is a list of indices in 0..6560
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
    
    # Coverage count: how many codewords cover each word
    coverage = np.zeros(6561, dtype=np.int32)
    for neigh in neighborhoods:
        for idx in neigh:
            coverage[idx] += 1
    
    # Peeling loop with improved redundancy scoring
    # We use a more sophisticated criterion: prioritize removal of codewords
    # whose neighborhood has the highest minimum coverage (most redundant)
    changed = True
    while changed:
        changed = False
        
        best_idx = -1
        best_min_coverage = 0
        
        for i in range(m):
            # Check if codeword i is removable: all neighbors must have coverage >= 2
            min_cov = 17  # maximum possible
            for idx in neighborhoods[i]:
                cov = coverage[idx]
                if cov < 2:
                    min_cov = 0
                    break
                if cov < min_cov:
                    min_cov = cov
            
            if min_cov > best_min_coverage:
                best_min_coverage = min_cov
                best_idx = i
        
        if best_idx >= 0:
            # Remove codeword best_idx
            for idx in neighborhoods[best_idx]:
                coverage[idx] -= 1
            # Remove from code_arr and neighborhoods
            code_arr = np.delete(code_arr, best_idx, axis=0)
            neighborhoods.pop(best_idx)
            m -= 1
            changed = True
    
    return [list(row) for row in code_arr]
# EVOLVE-BLOCK-END
