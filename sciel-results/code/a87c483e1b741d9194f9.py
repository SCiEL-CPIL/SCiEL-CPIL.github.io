import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s buffer for safety
    
    # Pre-compute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Pre-compute all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, store the list of t-subset indices it covers
    # Use numpy arrays for faster indexing
    block_to_subs = []
    # Inverted index: for each t-subset, list of block indices that contain it
    sub_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        block_set = set(block)
        subs = []
        for s in all_t_subsets:
            if all(p in block_set for p in s):
                subs.append(t_subset_to_idx[s])
                sub_to_blocks[t_subset_to_idx[s]].append(b_idx)
        block_to_subs.append(np.array(subs, dtype=np.int32))
        
    best_blocks = None
    best_score = float('inf')
    
    for run in range(50):
        if time.time() > deadline:
            break
            
        # Greedy Construction
        coverage_count = np.zeros(num_t_subsets, dtype=np.int32)
        blocks = []
        uncovered_count = num_t_subsets
        
        while uncovered_count > 0:
            if time.time() > deadline:
                break
            
            # Pick a random uncovered t-subset
            # Find an index with 0 count
            # Linear search is fast enough for 2024 elements
            uncovered_indices = [i for i in range(num_t_subsets) if coverage_count[i] == 0]
            if not uncovered_indices:
                break
            target_idx = random.choice(uncovered_indices)
            
            # Get all blocks containing this t-subset
            candidate_block_indices = sub_to_blocks[target_idx]
            
            best_block_idx = -1
            best_gain = -1
            
            for b_idx in candidate_block_indices:
                subs = block_to_subs[b_idx]
                # Calculate gain: number of subs in this block that are currently uncovered
                gain = np.sum(coverage_count[subs] == 0)
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
                    if best_gain == len(subs): # Max possible gain for this block
                        break
            
            if best_block_idx == -1:
                break
                
            # Add block
            blocks.append(best_block_idx)
            subs = block_to_subs[best_block_idx]
            
            # Update coverage
            for s in subs:
                if coverage_count[s] == 0:
                    uncovered_count -= 1
                coverage_count[s] += 1
        
        if uncovered_count > 0:
            continue
            
        # Post-processing: Remove redundant blocks
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                subs = block_to_subs[b_idx]
                # Check if all subs are covered by at least 1 other block
                if np.all(coverage_count[subs] > 1):
                    for s in subs:
                        coverage_count[s] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
            
            if time.time() > deadline:
                break
        
        # Local Search: Swap to allow removal
        local_search_time_limit = 2.0
        ls_start = time.time()
        
        improved = True
        while improved and (time.time() - ls_start) < local_search_time_limit:
            improved = False
            if time.time() > deadline:
                break
                
            # Try swapping each block
            for i in range(len(blocks)):
                if time.time() > deadline:
                    break
                    
                b_old = blocks[i]
                subs_old = block_to_subs[b_old]
                
                # Tentatively remove
                for s in subs_old:
                    coverage_count[s] -= 1
                
                # Find uncovered subsets (count == 0)
                uncovered = [s for s in subs_old if coverage_count[s] == 0]
                
                if not uncovered:
                    # Block was redundant
                    blocks.pop(i)
                    improved = True
                    break
                else:
                    # Need to find a new block that covers all 'uncovered' subsets
                    # Intersection of candidate blocks for each uncovered subset
                    min_list_len = float('inf')
                    start_s = -1
                    for s in uncovered:
                        if len(sub_to_blocks[s]) < min_list_len:
                            min_list_len = len(sub_to_blocks[s])
                            start_s = s
                            
                    candidates = set(sub_to_blocks[start_s])
                    for s in uncovered:
                        if s == start_s: continue
                        candidates.intersection_update(sub_to_blocks[s])
                        
                    if not candidates:
                        # Restore counts
                        for s in subs_old:
                            coverage_count[s] += 1
                        continue
                        
                    # Pick a candidate
                    # Randomly pick one to diversify
                    b_new = random.choice(list(candidates))
                    
                    if b_new == b_old:
                        # If we pick the same block, it's a no-op swap.
                        # But we removed it, so we must add a different one?
                        # If b_new == b_old, it means the old block was the only one covering 'uncovered'.
                        # But we removed it, so 'uncovered' are 0.
                        # If we add it back, we are back to start.
                        # So we must pick a different block if possible.
                        if len(candidates) > 1:
                            # Pick another one
                            candidates_list = list(candidates)
                            candidates_list.remove(b_old)
                            b_new = random.choice(candidates_list)
                        else:
                            # Only option is the old block. Skip.
                            for s in subs_old:
                                coverage_count[s] += 1
                            continue

                    # Perform swap
                    blocks[i] = b_new
                    subs_new = block_to_subs[b_new]
                    for s in subs_new:
                        coverage_count[s] += 1
                    improved = True
                    break # Restart loop with new state
        
        # Final removal pass
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for b_idx in blocks:
                subs = block_to_subs[b_idx]
                if np.all(coverage_count[subs] > 1):
                    for s in subs:
                        coverage_count[s] -= 1
                    changed = True
                else:
                    new_blocks.append(b_idx)
            blocks = new_blocks
        
        if len(blocks) < best_score:
            best_score = len(blocks)
            best_blocks = blocks.copy()
            
    if best_blocks is None:
        return []
        
    # Convert block indices back to lists of points
    final_blocks = [list(all_blocks[i]) for i in best_blocks]
    return final_blocks
