import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    # 1. Precomputation
    # Generate all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # For each block, store the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # For each t-subset, store the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)

    # 2. Greedy Set Cover
    uncovered = [True] * num_t_subsets
    gain = [0] * num_k_subsets
    
    # Initialize gain
    for b_idx in range(num_k_subsets):
        gain[b_idx] = len(block_t_indices[b_idx])
        
    selected_blocks = []
    
    while True:
        max_gain = max(gain)
        if max_gain == 0:
            break
            
        # Find the block with max gain
        best_block = gain.index(max_gain)
        
        selected_blocks.append(best_block)
        
        # Update gains
        for t_idx in block_t_indices[best_block]:
            if uncovered[t_idx]:
                uncovered[t_idx] = False
                for b_idx in t_subset_blocks[t_idx]:
                    gain[b_idx] -= 1
                    
    # 3. Local Search: Deletion and Swapping
    # Calculate cover counts
    cover_count = [0] * num_t_subsets
    for b_idx in selected_blocks:
        for t_idx in block_t_indices[b_idx]:
            cover_count[t_idx] += 1
            
    active_blocks = set(selected_blocks)
    
    # Helper to check if a block can be removed
    def can_remove(b_idx):
        for t_idx in block_t_indices[b_idx]:
            if cover_count[t_idx] <= 1:
                return False
        return True

    # Helper to find a swap
    def find_swap(b_idx):
        # Identify subsets in b_idx that are uniquely covered by b_idx
        critical_subsets = [t_idx for t_idx in block_t_indices[b_idx] if cover_count[t_idx] == 1]
        if not critical_subsets:
            return None # Can be removed, not swapped
        
        # Find candidates: blocks not in active_blocks that cover all critical_subsets
        # Intersect the block lists of all critical subsets
        candidates = set(t_subset_blocks[critical_subsets[0]])
        for s in critical_subsets[1:]:
            candidates.intersection_update(t_subset_blocks[s])
            
        # Remove blocks already in active set
        candidates.difference_update(active_blocks)
        
        if not candidates:
            return None
            
        # Score candidates: prefer those that cover more subsets already covered by others (increasing redundancy)
        best_candidate = None
        best_score = -1
        for cand in candidates:
            score = 0
            for t_idx in block_t_indices[cand]:
                if cover_count[t_idx] > 0:
                    score += 1
            if score > best_score:
                best_score = score
                best_candidate = cand
                
        return best_candidate

    # Iterate local search
    improved = True
    while improved:
        improved = False
        indices_to_check = list(active_blocks)
        random.shuffle(indices_to_check)
        
        for b_idx in indices_to_check:
            if b_idx not in active_blocks:
                continue
                
            # Try removal
            if can_remove(b_idx):
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                improved = True
                continue # Restart loop? Or continue? Continuing is faster, but might miss cascades.
                         # Given the shuffle, we'll catch it in the next pass.
            
            # Try swap
            swap_candidate = find_swap(b_idx)
            if swap_candidate is not None:
                # Perform swap
                active_blocks.remove(b_idx)
                for t_idx in block_t_indices[b_idx]:
                    cover_count[t_idx] -= 1
                    
                active_blocks.add(swap_candidate)
                for t_idx in block_t_indices[swap_candidate]:
                    cover_count[t_idx] += 1
                    
                improved = True

    # Convert back to list of lists
    result = [list(all_k_subsets[b_idx]) for b_idx in active_blocks]
    
    return result
