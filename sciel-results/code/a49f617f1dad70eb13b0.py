import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks)
    # For v=24, k=7, there are 346,104 blocks.
    # This might be memory intensive if we store all indices for all blocks.
    # 346,104 * 35 * 4 bytes (int32) ~ 48 MB. This is fine.
    
    blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(blocks)
    
    # Precompute block_indices: shape (num_blocks, t_subsets_per_block)
    # t_subsets_per_block = C(k, t) = C(7, 3) = 35
    block_indices = np.empty((num_blocks, 35), dtype=np.int32)
    for i, block in enumerate(blocks):
        for j, t_sub in enumerate(itertools.combinations(block, t)):
            block_indices[i, j] = t_subset_to_idx[t_sub]
            
    # Precompute frequency of each t-subset
    # Count how many blocks cover each t-subset
    freq = np.zeros(num_t_subsets, dtype=np.int32)
    # We can use np.bincount on the flattened block_indices
    freq = np.bincount(block_indices.flatten(), minlength=num_t_subsets)
    
    # Inverse frequency for weighting
    # Avoid division by zero, though freq should be > 0 for all
    inv_freq = 1.0 / (freq + 1e-9)
    
    uncovered = np.ones(num_t_subsets, dtype=bool)
    selected_blocks = []
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Calculate weighted gains
        # weights for current step: 1.0 if uncovered and weighted by inv_freq, else 0.0
        # We can precompute weighted_indices? No, uncovered changes.
        # gains = sum over j of (uncovered[idx] * inv_freq[idx])
        
        # Vectorized calculation:
        # Create a weighted array of size num_t_subsets
        # w = np.where(uncovered, inv_freq, 0.0)
        # gains = np.sum(w[block_indices], axis=1)
        
        w = np.where(uncovered, inv_freq, 0.0)
        gains = np.sum(w[block_indices], axis=1)
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            # Should not happen if a covering exists
            break
            
        selected_blocks.append(list(blocks[best_idx]))
        
        # Update uncovered
        uncovered[block_indices[best_idx]] = False
        
    return selected_blocks
