import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using Greedy + Local Search (Deletion + Swap).
    """
    # 1. Precomputation
    all_t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {ts: i for i, ts in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Precompute t-subset indices for each block
    # Shape: (num_blocks, C(k, t))
    c_k_t = int(len(list(itertools.combinations(range(k), t))))
    block_t_indices = np.zeros((num_blocks, c_k_t), dtype=np.int32)
    for i, block in enumerate(all_blocks):
        for j, ts in enumerate(itertools.combinations(block, t)):
             block_t_indices[i, j] = t_sub_to_idx[ts]
             
    # Precompute t_sub_to_blocks: for each t-subset, list of block indices covering it
    # This is the transpose of the incidence structure
    t_sub_to_blocks = [[] for _ in range(num_t_subs)]
    for i in range(num_blocks):
        for idx in block_t_indices[i]:
            t_sub_to_blocks[idx].append(i)
            
    # 2. Greedy Construction
    uncovered = np.ones(num_t_subs, dtype=np.int8)
    blocks_used = []
    used_block_mask = np.zeros(num_blocks, dtype=bool)
    
    while uncovered.sum() > 0:
        # Calculate gains
        gains = uncovered[block_t_indices].sum(axis=1)
        gains[used_block_mask] = -1
        best_idx = np.argmax(gains)
        
        if gains[best_idx] <= 0:
            break
            
        blocks_used.append(best_idx) # Store indices for efficiency
        used_block_mask[best_idx] = True
        uncovered[block_t_indices[best_idx]] = 0
        
    # 3. Local Search
    # Initialize coverage counts
    coverage_count = np.zeros(num_t_subs, dtype=np.int32)
    for b_idx in blocks_used:
        coverage_count[block_t_indices[b_idx]] += 1
        
    # Helper to check if a block can be deleted
    def can_delete(b_idx):
        idxs = block_t_indices[b_idx]
        # If any t-subset in this block has count 1, it's critical
        return np.min(coverage_count[idxs]) > 1

    # Helper to find replacement candidates
    def find_replacements(b_idx):
        idxs = block_t_indices[b_idx]
        # Find t-subsets covered only by this block
        unique_mask = (coverage_count[idxs] == 1)
        if not np.any(unique_mask):
            return [] # Can be deleted
        
        unique_idxs = idxs[unique_mask]
        
        # Intersection of t_sub_to_blocks for all unique t-subsets
        # Start with the first one's list
        if len(unique_idxs) == 0:
            return []
            
        candidates = set(t_sub_to_blocks[unique_idxs[0]])
        candidates.discard(b_idx)
        
        for u_idx in unique_idxs[1:]:
            candidates.intersection_update(t_sub_to_blocks[u_idx])
            
        return list(candidates)

    # Local Search Loop
    improved = True
    while improved:
        improved = False
        
        # Phase 1: Deletion
        for i in range(len(blocks_used) - 1, -1, -1):
            b_idx = blocks_used[i]
            if can_delete(b_idx):
                # Remove block
                coverage_count[block_t_indices[b_idx]] -= 1
                blocks_used.pop(i)
                improved = True
                break # Restart loop
        
        if improved:
            continue
            
        # Phase 2: Swap
        # Try to swap a block to enable further deletions
        # To avoid getting stuck, we can randomize the order or try all
        for i in range(len(blocks_used)):
            b_idx = blocks_used[i]
            candidates = find_replacements(b_idx)
            
            if candidates:
                # Select a candidate
                # Heuristic: Choose the one that covers the most "redundant" t-subsets?
                # Or just pick one randomly to explore different local optima
                # Let's pick the one that maximizes the sum of coverage counts of the t-subsets it covers
                # This might help in making other blocks redundant?
                # Actually, if we replace B with B', the coverage counts of t-subsets in B' increase,
                # and counts in B decrease.
                # We want to increase redundancy elsewhere.
                
                best_candidate = None
                best_score = -1
                
                for c_idx in candidates:
                    # Score: How many t-subsets does c_idx cover that are already covered by >1 block?
                    # This is a proxy for "how much redundancy we add".
                    c_idxs = block_t_indices[c_idx]
                    # We want to add coverage to places where it's already safe
                    # Or, simpler: just pick the first one to be fast
                    # Let's try a simple heuristic: pick the candidate that overlaps most with current blocks?
                    # For speed, let's just pick a random candidate or the first one.
                    # Randomness helps escape local minima.
                    best_candidate = random.choice(candidates)
                    break
                
                if best_candidate is not None and best_candidate != b_idx:
                    # Perform swap
                    # Decrease counts for b_idx
                    coverage_count[block_t_indices[b_idx]] -= 1
                    # Increase counts for best_candidate
                    coverage_count[block_t_indices[best_candidate]] += 1
                    # Update blocks_used
                    blocks_used[i] = best_candidate
                    improved = True
                    break # Restart deletion phase

    # Convert block indices to lists of integers
    blocks = [list(all_blocks[b_idx]) for b_idx in blocks_used]
    
    return blocks
