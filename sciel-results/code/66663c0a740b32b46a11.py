import itertools
import numpy as np
import time
import random
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # 1. Precomputation
    points = list(range(v))
    all_t_subs = list(itertools.combinations(points, t))
    t_sub_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    num_t_subs = len(all_t_subs)
    
    all_k_subs = list(itertools.combinations(points, k))
    num_k_subs = len(all_k_subs)
    
    # Precompute coverage for each k-subset
    # covers[i] is a numpy array of t-subset indices covered by block i
    covers = []
    for b in all_k_subs:
        b_set = set(b)
        ts_indices = [t_sub_idx[ts] for ts in itertools.combinations(b, t)]
        covers.append(np.array(ts_indices, dtype=np.int32))
        
    # Map each t-subset to the indices of k-subsets that cover it
    # ts_to_blocks[j] is a numpy array of k-subset indices
    ts_to_blocks = [[] for _ in range(num_t_subs)]
    for i, b in enumerate(all_k_subs):
        for ts_idx in covers[i]:
            ts_to_blocks[ts_idx].append(i)
    for i in range(num_t_subs):
        ts_to_blocks[i] = np.array(ts_to_blocks[i], dtype=np.int32)
    
    best_blocks = None
    best_size = float('inf')
    
    # 2. Greedy Phase (Multiple Seeds)
    num_greedy_runs = 10
    for run in range(num_greedy_runs):
        if time.time() > deadline - 30: break
        
        random.seed(run * 1000 + 42)
        cover_count = np.zeros(num_t_subs, dtype=np.int32)
        selected_blocks = []
        selected_indices = set()
        
        while True:
            if time.time() > deadline - 20: break
            uncovered_mask = cover_count == 0
            if not np.any(uncovered_mask):
                break
            
            # Find best block using bottleneck heuristic:
            # Prioritize blocks that cover the most "hard to cover" subsets
            # We use a simple greedy: maximize gain (newly covered subsets)
            # But we can also weight by scarcity: subsets with fewer candidate blocks are "harder"
            
            best_gain = -1
            best_block_idx = -1
            
            # To speed up, we can precompute for each block the list of t-subsets it covers
            # and just sum the uncovered mask over those indices.
            
            for i in range(num_k_subs):
                if i in selected_indices:
                    continue
                # Gain is number of uncovered subsets covered by block i
                mask = uncovered_mask[covers[i]]
                gain = np.sum(mask)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
                    if gain == num_t_subs: # Perfect block? Unlikely
                        break
            
            if best_block_idx == -1:
                break # Should not happen if uncovered exists
                
            selected_blocks.append(all_k_subs[best_block_idx])
            selected_indices.add(best_block_idx)
            cover_count[covers[best_block_idx]] += 1
            
        if len(selected_blocks) < best_size:
            best_size = len(selected_blocks)
            best_blocks = [list(b) for b in selected_blocks]
    
    if best_blocks is None:
        # Fallback to simple greedy if time ran out
        return construct_fallback(v, k, t)

    # 3. Local Search / Refinement
    # Work with indices of blocks for efficiency
    # Map block tuple to index
    block_to_idx = {b: i for i, b in enumerate(all_k_subs)}
    current_blocks_idx = [block_to_idx[tuple(b)] for b in best_blocks]
    
    # Re-compute cover counts for the current best solution
    cover_count = np.zeros(num_t_subs, dtype=np.int32)
    for i in current_blocks_idx:
        cover_count[covers[i]] += 1
        
    def try_remove_redundant(blocks_list_idx, cover_counts):
        removed = False
        # Shuffle order to break symmetry
        indices = list(range(len(blocks_list_idx)))
        random.shuffle(indices)
        
        for i in indices:
            if time.time() > deadline - 5:
                return False, blocks_list_idx, cover_counts
            
            b_idx = blocks_list_idx[i]
            # Check if removing b leaves any 0s
            # Decrement
            cover_counts[covers[b_idx]] -= 1
            
            if np.min(cover_counts) > 0:
                # Remove
                blocks_list_idx.pop(i)
                removed = True
                # Restart to find more
                return True, blocks_list_idx, cover_counts
            else:
                # Restore
                cover_counts[covers[b_idx]] += 1
        return removed, blocks_list_idx, cover_counts

    # Run redundancy removal
    improved = True
    while improved:
        if time.time() > deadline - 5: break
        improved, current_blocks_idx, cover_count = try_remove_redundant(current_blocks_idx, cover_count)

    # Block Replacement Search
    # Precompute ts_to_blocks as sets for faster intersection?
    # ts_to_blocks is list of arrays.
    
    def find_replacement(b_old_idx, blocks_list_idx, cover_counts):
        # Find private subsets
        private_ts = []
        for ts_idx in covers[b_old_idx]:
            if cover_counts[ts_idx] == 1:
                private_ts.append(ts_idx)
                
        if not private_ts:
            return b_old_idx, True # Can remove
            
        # Find candidates that cover all private_ts
        # Intersect ts_to_blocks for each private ts
        candidates = set(ts_to_blocks[private_ts[0]].tolist())
        for ts_idx in private_ts[1:]:
            candidates &= set(ts_to_blocks[ts_idx].tolist())
            
        if not candidates:
            return b_old_idx, False
            
        # Pick a candidate (random or heuristic)
        # Heuristic: Maximize overlap with existing blocks?
        # For now, pick random
        cand_idx = random.choice(list(candidates))
        
        # Check if cand_idx is already in blocks_list
        # If yes, it's a duplicate, which is bad (blocks must be distinct)
        # We need to ensure distinctness.
        # If cand_idx is already there, we can't just "replace" with the same one.
        # We need a *different* block.
        # Filter out existing blocks
        existing_set = set(blocks_list_idx)
        valid_cands = [c for c in candidates if c not in existing_set]
        
        if not valid_cands:
            return b_old_idx, False
            
        cand_idx = random.choice(valid_cands)
        return cand_idx, False # Replaced, not removed

    # Local Search Loop
    max_iter = 5000
    for _ in range(max_iter):
        if time.time() > deadline - 5:
            break
            
        # Randomly pick a block to modify
        i = random.randint(0, len(current_blocks_idx) - 1)
        b_old_idx = current_blocks_idx[i]
        
        # Try to find replacement
        # First, check if it can be removed
        # (Simplified: use the logic from above)
        
        # Re-calculate private subsets
        private_ts = []
        for ts_idx in covers[b_old_idx]:
            if cover_count[ts_idx] == 1:
                private_ts.append(ts_idx)
                
        if not private_ts:
            # Remove it
            # Decrement counts
            cover_count[covers[b_old_idx]] -= 1
            current_blocks_idx.pop(i)
            # Run redundancy removal again
            improved, current_blocks_idx, cover_count = try_remove_redundant(current_blocks_idx, cover_count)
            continue
            
        # Find candidates
        candidates = set(ts_to_blocks[private_ts[0]].tolist())
        for ts_idx in private_ts[1:]:
            candidates &= set(ts_to_blocks[ts_idx].tolist())
            
        # Exclude current block and any already in list
        existing_set = set(current_blocks_idx)
        candidates.discard(b_old_idx)
        candidates -= existing_set
        
        if candidates:
            cand_idx = random.choice(list(candidates))
            # Replace
            # Decrement counts for b_old
            cover_count[covers[b_old_idx]] -= 1
            # Increment counts for cand
            cover_count[covers[cand_idx]] += 1
            current_blocks_idx[i] = cand_idx
            
            # Try to remove redundancies
            improved, current_blocks_idx, cover_count = try_remove_redundant(current_blocks_idx, cover_count)
            # If size didn't decrease, we might want to revert?
            # Hill climbing: only keep if size <= previous.
            # Since we just did one move, if size is same, we can keep it to explore further.
            # If size increased (impossible here as we replaced 1 with 1), revert.
            
    return [list(b) for b in current_blocks_idx]

def construct_fallback(v, k, t):
    # Simple greedy if main fails
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    while uncovered:
        block = list(min(uncovered))
        while len(block) < k:
            def gain(x):
                return sum(1 for rest in itertools.combinations(block, t - 1)
                           if tuple(sorted(rest + (x,))) in uncovered)
            block.append(max((x for x in range(v) if x not in block), key=gain))
        block.sort()
        blocks.append(block)
        uncovered.difference_update(itertools.combinations(block, t))
    return blocks
