<<<<<<< SEARCH
def _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k):
    """Greedy set cover using NumPy vectorization."""
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    
    while True:
        if _elapsed() > _DEADLINE - 25:
            break
        
        # Calculate gain for each block
        # cov_matrix is uint8, uncovered is uint8. Product is 1 if covered and uncovered.
        gains = np.sum(cov_matrix * uncovered, axis=1)
        
        best_i = np.argmax(gains)
        best_gain = gains[best_i]
        
        if best_gain == 0:
            break
            
        blocks.append(best_i)
        uncovered[cov_matrix[best_i] == 1] = 0
        
        if np.sum(uncovered) == 0:
            break
            
    return blocks
=======
def _greedy_construct(v, k, t, rng, cov_matrix, n_t, n_k):
    """Packing-First greedy: minimize pairwise overlap while covering all t-subsets.
    
    Uses a composite score that balances coverage gain against overlap penalty,
    encouraging selection of blocks that share few t-subsets with already-chosen blocks.
    """
    uncovered = np.ones(n_t, dtype=np.uint8)
    blocks = []
    
    # Track which t-subsets are covered by how many selected blocks
    # For overlap penalty: count how many times each t-subset is covered
    covered_counts = np.zeros(n_t, dtype=np.int32)
    
    while True:
        if _elapsed() > _DEADLINE - 20:
            break
        
        # Calculate pure coverage gain
        gains = np.sum(cov_matrix * uncovered, axis=1)
        
        best_gain = np.max(gains)
        if best_gain == 0:
            break
            
        # Find all blocks with maximum gain
        max_gain_blocks = np.where(gains == best_gain)[0]
        
        if len(max_gain_blocks) == 1:
            best_i = max_gain_blocks[0]
        else:
            # Among max-gain blocks, pick one with minimum overlap with already-covered t-subsets
            # Overlap = number of t-subsets in this block that are already covered (count > 0)
            overlaps = np.sum(covered_counts[max_gain_blocks] > 0, axis=1)
            min_overlap = np.min(overlaps)
            candidates = max_gain_blocks[overlaps == min_overlap]
            # Random tie-break among minimum-overlap candidates
            best_i = candidates[rng.randrange(len(candidates))]
            
        blocks.append(best_i)
        
        # Update coverage
        block_cov = cov_matrix[best_i]
        uncovered[block_cov == 1] = 0
        covered_counts[block_cov == 1] += 1
        
        if np.sum(uncovered) == 0:
            break
            
    return blocks
>>>>>>> REPLACE

