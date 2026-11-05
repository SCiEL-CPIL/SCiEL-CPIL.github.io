import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # 1. Precomputation
    # All t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # All k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute covered indices for each block
    # Also build the dense matrix M for fast greedy
    # M shape: (n_blocks, n_subsets)
    # Using uint8 to save memory
    M = np.zeros((n_blocks, n_subsets), dtype=np.uint8)
    covered_ids = []
    
    for i, block in enumerate(all_blocks):
        idxs = []
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            M[i, idx] = 1
            idxs.append(idx)
        covered_ids.append(np.array(idxs, dtype=np.int32))
        
    # Build reverse index: subset_idx -> list of block indices
    subset_to_blocks = [[] for _ in range(n_subsets)]
    for i in range(n_blocks):
        for idx in covered_ids[i]:
            subset_to_blocks[idx].append(i)
            
    # 2. Greedy Initialization
    # uncovered: 1 if subset is not covered yet, 0 otherwise
    uncovered = np.ones(n_subsets, dtype=float)
    selected = np.zeros(n_blocks, dtype=bool)
    selected_indices = []
    
    # cover_counts for local search
    cover_counts = np.zeros(n_subsets, dtype=np.int32)
    
    while time.time() < deadline:
        if np.all(uncovered == 0):
            break
            
        # Calculate gains: number of uncovered subsets covered by each block
        # M is (n_blocks, n_subsets), uncovered is (n_subsets,)
        gains = M.dot(uncovered)
        
        # Mask out already selected blocks
        gains[selected] = -1
        
        best_block = np.argmax(gains)
        best_gain = gains[best_block]
        
        if best_gain <= 0:
            break # No block can cover any new subset
            
        # Select the block
        selected[best_block] = True
        selected_indices.append(best_block)
        
        # Update uncovered and cover_counts
        idxs = covered_ids[best_block]
        uncovered[idxs] = 0
        cover_counts[idxs] += 1

    best_size = len(selected_indices)
    best_blocks_set = set(selected_indices)
    
    # 3. Local Search
    random.seed(42)
    
    # Helper to check time
    def check_time():
        return time.time() > deadline

    # Removal Phase
    while time.time() < deadline:
        removed = False
        # Shuffle selected blocks to try different order
        current_list = list(best_blocks_set)
        random.shuffle(current_list)
        
        for b in current_list:
            if check_time():
                break
                
            # Check if b is redundant
            # Redundant if all subsets it covers have count > 1
            idxs_b = covered_ids[b]
            if np.all(cover_counts[idxs_b] > 1):
                # Remove b
                cover_counts[idxs_b] -= 1
                best_blocks_set.remove(b)
                removed = True
                best_size = len(best_blocks_set)
                break # Restart removal search after a change
        
        if not removed:
            break

    # Swap Phase (1-for-1) to escape local minima
    # We try to swap a block b1 with b2 such that size remains same but structure improves
    # "Improves" is hard to define simply. We just try to find valid swaps.
    # If we find a swap, we do it. We limit the number of swaps to avoid infinite loops.
    
    max_swaps = 50
    swaps_done = 0
    
    while time.time() < deadline and swaps_done < max_swaps:
        swapped = False
        current_list = list(best_blocks_set)
        random.shuffle(current_list)
        
        for b1 in current_list:
            if check_time() or swapped:
                break
                
            idxs_b1 = covered_ids[b1]
            
            # Identify critical subsets: those with count == 1 in b1
            # If we remove b1, these become 0 (uncovered)
            critical_idxs = idxs_b1[cover_counts[idxs_b1] == 1]
            
            if len(critical_idxs) == 0:
                # This block is redundant, should have been removed in previous phase
                # But maybe counts changed? No, we break on removal.
                # If len==0, it means all are >1, so it's redundant.
                # We should remove it.
                cover_counts[idxs_b1] -= 1
                best_blocks_set.remove(b1)
                best_size = len(best_blocks_set)
                swapped = True
                break
                
            if len(critical_idxs) > 3:
                # Hard to find a single block covering >3 specific subsets
                # Skip this b1
                continue
                
            # Find a candidate b2
            # b2 must cover all critical_idxs
            # Start with the list of blocks covering the first critical subset
            c0 = critical_idxs[0]
            candidates = subset_to_blocks[c0]
            
            # Filter candidates:
            # 1. Not in best_blocks_set
            # 2. Covers all other critical_idxs
            
            valid_b2 = -1
            for b2 in candidates:
                if b2 in best_blocks_set:
                    continue
                
                # Check if b2 covers all critical_idxs
                # critical_idxs is small (1 to 3)
                # Use np.isin or set intersection
                # covered_ids[b2] is an array
                if len(critical_idxs) == 1:
                    valid_b2 = b2
                    break
                else:
                    # Check if all critical_idxs are in covered_ids[b2]
                    # np.all(np.isin(critical_idxs, covered_ids[b2]))
                    # This is fast for small arrays
                    if np.all(np.isin(critical_idxs, covered_ids[b2])):
                        valid_b2 = b2
                        break
                        
            if valid_b2 != -1:
                # Perform swap
                # Remove b1
                cover_counts[idxs_b1] -= 1
                best_blocks_set.remove(b1)
                
                # Add b2
                idxs_b2 = covered_ids[valid_b2]
                cover_counts[idxs_b2] += 1
                best_blocks_set.add(valid_b2)
                
                swapped = True
                swaps_done += 1
                break # Restart swap search after a change

    # Convert best_blocks to list of lists
    return [list(all_blocks[i]) for i in sorted(best_blocks_set)]
