"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import itertools
import time
import random

D = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    # Keep a safety margin of 10 seconds
    deadline = start_time + 110.0
    
    # Generate all possible 4-subsets of coordinates
    all_subsets = list(itertools.combinations(range(D), 4))
    num_subsets = len(all_subsets)
    
    # Pre-compute the 3-subsets (triples) contained in each 4-subset
    # Two 4-subsets conflict if they share a triple (intersection size >= 3)
    subset_triples = []
    for s in all_subsets:
        triples = list(itertools.combinations(s, 3))
        subset_triples.append(triples)
        
    best_blocks = []
    best_len = 0
    
    # Use a fixed seed for reproducibility of the search process itself,
    # but the search is randomized.
    rng = np.random.default_rng(42)
    
    # Theoretical maximum number of blocks is floor(165/4) = 41
    max_possible_blocks = 41
    
    # Randomized Greedy Search
    while time.time() < deadline:
        # Shuffle the order of subsets to try
        indices = np.arange(num_subsets)
        rng.shuffle(indices)
        
        used_triples = set()
        current_blocks = []
        
        for idx in indices:
            triples = subset_triples[idx]
            # Check for conflict: any triple already used?
            conflict = False
            for t in triples:
                if t in used_triples:
                    conflict = True
                    break
            
            if not conflict:
                current_blocks.append(all_subsets[idx])
                for t in triples:
                    used_triples.add(t)
                    
        if len(current_blocks) > best_len:
            best_len = len(current_blocks)
            best_blocks = current_blocks
            
            if best_len >= max_possible_blocks:
                break

    # If no blocks found (should not happen), fallback to a single block
    if not best_blocks:
        best_blocks = [all_subsets[0]]

    # Construct the vectors
    # For each block (4-subset), generate all 16 sign variations
    vectors = []
    for block in best_blocks:
        # itertools.product generates tuples of (1, -1)
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for i, s in zip(block, signs):
                v[i] = s
            vectors.append(v)
            
    return np.array(vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
