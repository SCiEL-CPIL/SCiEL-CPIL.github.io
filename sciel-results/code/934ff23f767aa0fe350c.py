import itertools
import random
import numpy as np
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Precomputed Incidence Structure
    with Greedy Initialization and Randomized Local Search.
    """
    start_time = time.time()
    time_limit = 115.0

    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Precompute block -> t-subset indices mapping
    # block_t_indices[i] is a list of indices of t-subsets covered by all_k_subsets[i]
    block_t_indices = []
    for block in all_k_subsets:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_to_idx[ts])
        block_t_indices.append(indices)
        
    # Precompute t-subset -> block indices mapping
    # t_to_blocks[j] is a list of indices of blocks covering all_t_subsets[j]
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different random seeds for the local search
    max_restarts = 50
    
    for seed in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        
        # --- Phase 1: Greedy Construction ---
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            t_idx = random.choice(tuple(uncovered))
            
            # Get all blocks that cover this t-subset
            candidates = t_to_blocks[t_idx]
            
            best_b_idx = -1
            best_gain = -1
            
            # Evaluate candidates
            # Optimization: If a candidate covers many uncovered, it's good.
            for b_idx in candidates:
                gain = 0
                # Count how many t-subsets in this block are currently uncovered
                for t2_idx in block_t_indices[b_idx]:
                    if t2_idx in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination if we can't beat best? 
                            # Max possible gain is len(block_t_indices[b_idx])
                            # If best_gain is already high, we can skip?
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            # Add the best block
            current_blocks.append(best_b_idx)
            for t2_idx in block_t_indices[best_b_idx]:
                coverage[t2_idx] += 1
                if coverage[t2_idx] == 1:
                    uncovered.discard(t2_idx)
                    
        # --- Phase 2: Redundancy Removal (Cleanup) ---
        def remove_redundant(blocks, coverage):
            """
            Removes redundant blocks from the list.
            Returns the new list of block indices and updated coverage.
            """
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    b_idx = blocks[i]
                    is_redundant = True
                    # Check if all t-subsets in this block are covered by > 1 block
                    for t_idx in block_t_indices[b_idx]:
                        if coverage[t_idx] <= 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Remove block
                        blocks.pop(i)
                        for t_idx in block_t_indices[b_idx]:
                            coverage[t_idx] -= 1
                        changed = True
                        break # Restart the loop to check for new redundancies
            return blocks, coverage

        current_blocks, coverage = remove_redundant(current_blocks, coverage)
        
        # Update best
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]

        # --- Phase 3: Randomized Local Search ---
        # We try to swap blocks to find a configuration that allows more removals.
        # Or simply swap to maintain validity and hope for a better structure.
        
        # To speed up, we limit the number of swap attempts per restart
        max_swaps = 2000
        swaps_attempted = 0
        
        while swaps_attempted < max_swaps:
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random block to replace
            if not current_blocks:
                break
            
            i = random.randint(0, len(current_blocks) - 1)
            old_b_idx = current_blocks[i]
            
            # Pick a random new block
            # It's better to pick a block that shares many t-subsets with the old one?
            # Or just random?
            # Let's pick a random block from all_k_subsets.
            new_b_idx = random.randint(0, num_k_subsets - 1)
            
            if old_b_idx == new_b_idx:
                continue
                
            # Check if swap is valid
            # A swap is valid if no t-subset coverage drops to 0.
            # Specifically, for any t-subset in old_b_idx that is NOT in new_b_idx,
            # coverage must be > 1.
            
            # Optimization: 
            # 1. Identify t-subsets in old_b_idx.
            # 2. Identify t-subsets in new_b_idx.
            # 3. For t in old \ new: check coverage[t] > 1.
            # 4. For t in new \ old: coverage[t] will increase, so always safe.
            
            old_t_set = set(block_t_indices[old_b_idx])
            new_t_set = set(block_t_indices[new_b_idx])
            
            valid = True
            # Check t-subsets that will lose coverage
            for t_idx in old_t_set:
                if t_idx not in new_t_set:
                    if coverage[t_idx] <= 1:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Perform swap
            # Decrement old
            for t_idx in old_t_set:
                coverage[t_idx] -= 1
            # Increment new
            for t_idx in new_t_set:
                coverage[t_idx] += 1
            
            current_blocks[i] = new_b_idx
            
            # Try to remove redundant blocks
            old_size = len(current_blocks)
            current_blocks, coverage = remove_redundant(current_blocks, coverage)
            new_size = len(current_blocks)
            
            # Acceptance criterion
            # If we reduced size, definitely accept (it's already in current_blocks).
            # If size is same, we might want to keep the new structure to explore further.
            # If size increased (shouldn't happen if remove_redundant is perfect, but maybe 
            # the swap itself didn't allow removal and we just swapped), we should revert?
            # Actually, if size didn't decrease, the swap didn't help immediately.
            # However, it might help later. 
            # To avoid wandering, let's only keep swaps that result in same or smaller size.
            # If smaller, great. If same, keep with 50% probability? Or always keep to change state?
            # Let's always keep if size <= old_size.
            if new_size > old_size:
                # Revert
                # This case is rare if remove_redundant is correct, but possible if 
                # the swap made the solution "worse" in terms of redundancy potential?
                # No, remove_redundant finds the maximal set of redundant blocks.
                # So new_size should be <= old_size if the swap was valid?
                # Wait. If I swap B1 with B2, and B2 is very redundant, I might remove B2 immediately?
                # No, I just swapped. Then I run remove_redundant.
                # If B2 was redundant, it gets removed. Size decreases.
                # If B1 was redundant, it was already removed? No, we picked a random block from current list.
                # So B1 is NOT redundant.
                # If B2 is redundant, size decreases.
                # If B2 is not redundant, size stays same (unless B1 became redundant due to swap? 
                # No, B1 is removed. B2 added. If B2 covers everything B1 did plus more, 
                # other blocks might become redundant.
                # So size should never increase.
                pass
            
            swaps_attempted += 1

            # Update best
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks[:]
                
    # Convert indices back to lists of integers
    final_blocks = [list(all_k_subsets[idx]) for idx in best_blocks]
    return final_blocks
