"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import itertools
import random

def construct(q, n, radius):
    # Generate all words in {0,...,q-1}^n
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    total = all_words.shape[0]  # q**n
    
    # For each word, precompute the set of words within Hamming distance <= radius
    # We'll store as a list of arrays or a 2D array of indices
    # Since radius is small (1 for this task), each word covers 1 + n*(q-1) words
    
    # Precompute coverage: for each word index, which word indices are covered
    coverage = []
    for i in range(total):
        w = all_words[i]
        covered = set()
        covered.add(i)
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val == orig:
                    continue
                w2 = w.copy()
                w2[pos] = val
                # Find index of w2
                idx = 0
                for j in range(n):
                    idx = idx * q + int(w2[j])
                covered.add(idx)
        coverage.append(np.array(sorted(covered), dtype=np.int32))
    
    # Convert to a single array for vectorized operations
    # All coverage sets have the same size for uniform radius
    cov_size = len(coverage[0])
    coverage_matrix = np.array(coverage, dtype=np.int32)  # shape (total, cov_size)
    
    # Greedy set cover
    uncovered = np.ones(total, dtype=bool)
    selected = []
    
    while uncovered.any():
        # For each candidate codeword, count how many uncovered words it covers
        counts = uncovered[coverage_matrix].sum(axis=1)
        best_idx = np.argmax(counts)
        
        if counts[best_idx] == 0:
            break
        
        selected.append(int(best_idx))
        uncovered[coverage_matrix[best_idx]] = False
    
    # Convert selected indices to list of codewords
    codewords = [all_words[i].tolist() for i in selected]
    
    return codewords
# EVOLVE-BLOCK-END
