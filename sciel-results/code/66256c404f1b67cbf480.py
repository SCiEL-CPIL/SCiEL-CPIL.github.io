"""Randomized greedy covering for C(19,7,4) using NumPy for speed."""
import itertools
import numpy as np
import time

# Global cache for precomputed data
_PRECOMPUTED = {}

def _get_precomputed(v, k, t):
    if (v, k, t) in _PRECOMPUTED:
        return _PRECOMPUTED[(v, k, t)]
        
    # Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # Precompute index matrix: for each block, indices of t-subsets it covers
    # Shape: (num_k_blocks, C(k, t))
    combos_in_block = list(itertools.combinations(range(k), t))
    num_combos = len(combos_in_block)
    
    index_matrix = np.zeros((num_k_blocks, num_combos), dtype=np.int32)
    for i, block in enumerate(k_blocks):
        for j, combo in enumerate(combos_in_block):
            # Map the relative combo to absolute points
            abs_combo = tuple(block[x] for x in combo)
            index_matrix[i, j] = t_subset_to_idx[abs_combo]
            
    _PRECOMPUTED[(v, k, t)] = (k_blocks, index_matrix, num_t_subsets)
    return _PRECOMPUTED[(v, k, t)]

def construct(v, k, t):
    start_time = time.time()
    time_limit = 110
    
    k_blocks, index_matrix, num_t_subsets = _get_precomputed(v, k, t)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple trials with different random seeds and alpha parameters
    # Alpha controls the "greediness". Higher alpha -> closer to max-gain greedy.
    # Lower alpha -> more uniform among candidates.
    alphas = [1.0, 1.5, 2.0, 0.5]
    trial = 0
    
    while time.time() - start_time < time_limit - 5:
        alpha = alphas[trial % len(alphas)]
        seed = trial * 12345 + 67890
        np.random.seed(seed)
        
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks_selected = []
        
        while uncovered.any():
            if time.time() - start_time > time_limit - 2:
                break
            
            # Calculate gains
            # uncovered[ index_matrix ] creates a boolean array of shape (num_k_blocks, num_combos)
            # sum(axis=1) gives the count of covered subsets for each block
            gains = uncovered[ index_matrix ].sum(axis=1)
            
            # Find candidates with gain > 0
            candidates = np.where(gains > 0)[0]
            
            if len(candidates) == 0:
                # Should not happen if a solution exists, but break to avoid infinite loop
                break
            
            # Calculate weights
            # Weights proportional to gain ** alpha
            weights = gains[candidates] ** alpha
            total_weight = weights.sum()
            
            if total_weight == 0:
                break
                
            probs = weights / total_weight
            
            # Select a block
            idx = np.random.choice(candidates, p=probs)
            
            # Update uncovered
            uncovered[ index_matrix[idx] ] = False
            
            # Record block
            blocks_selected.append(k_blocks[idx])
            
        if not uncovered.any() and len(blocks_selected) < best_count:
            best_count = len(blocks_selected)
            best_blocks = [list(b) for b in blocks_selected]
            
        trial += 1
        
    if best_blocks is None:
        # Fallback: simple deterministic greedy
        uncovered = np.ones(num_t_subsets, dtype=bool)
        blocks_selected = []
        while uncovered.any():
            gains = uncovered[ index_matrix ].sum(axis=1)
            candidates = np.where(gains > 0)[0]
            if len(candidates) == 0: break
            # Max gain
            idx = candidates[np.argmax(gains[candidates])]
            uncovered[ index_matrix[idx] ] = False
            blocks_selected.append(k_blocks[idx])
        best_blocks = [list(b) for b in blocks_selected]
        
    return best_blocks
