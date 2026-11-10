import itertools
import random
import time
import math

def construct(v, k, t):
    # Precomputation
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_id = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Precompute for each subset, the list of blocks that contain it
    # A block is defined by choosing k points from v
    # For v=19, k=7, number of blocks is C(19,7) = 50388
    # We can't precompute all, but we can precompute for each subset, the candidate blocks
    
    # For each t-subset, precompute the list of (k-t) subsets of the remaining points
    # that can be added to form a valid block
    subset_candidates = []
    for sub in all_subsets:
        sub_set = set(sub)
        remaining = [x for x in range(v) if x not in sub_set]
        num_add = k - t
        candidates = []
        for add_combo in itertools.combinations(remaining, num_add):
            block = tuple(sorted(sub + add_combo))
            candidates.append(block)
        subset_candidates.append(candidates)
    
    # Precompute for each block (as a frozenset or tuple), the list of t-subsets it covers
    # But we don't want to store all C(19,7) blocks. Instead, we'll compute on the fly or use a dict.
    # Actually, let's precompute a mapping from block tuple to list of subset ids
    # Number of distinct blocks in a covering is at most ~150. So we can just compute when needed.
    
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin
    
    best_blocks = None
    best_len = float('inf')
    
    runs = 0
    
    while time.time() < deadline:
        runs += 1
        # Reset state
        uncovered = bytearray([1] * n_subsets)
        uncovered_ids = list(range(n_subsets))
        id_to_idx = {i: i for i in range(n_subsets)}
        
        blocks = []
        block_set = set() # To avoid duplicate blocks
        
        while uncovered_ids:
            # Pick random uncovered subset
            idx = random.randrange(len(uncovered_ids))
            t_id = uncovered_ids[idx]
            target_subset = all_subsets[t_id]
            
            # Get precomputed candidates for this subset
            candidates = subset_candidates[t_id]
            
            # Calculate gain for each candidate
            best_gain = -1
            best_cands = []
            
            for block in candidates:
                if block in block_set:
                    continue
                # Calculate gain
                gain = 0
                # Iterate over all t-subsets of block
                for sub in itertools.combinations(block, t):
                    if uncovered[subset_to_id[sub]]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_cands = [block]
                elif gain == best_gain:
                    best_cands.append(block)
            
            # If all candidates are already in block_set, we have a problem
            # This shouldn't happen if the covering is valid, but let's handle it
            if not best_cands:
                # Fallback: pick any candidate (even if duplicate) - this indicates an issue
                # Actually, if all candidates are in block_set, it means we've already covered everything?
                # No, it means we're stuck. Let's just break.
                break
            
            # Pick one randomly from best candidates
            chosen_block = random.choice(best_cands)
            
            blocks.append(list(chosen_block))
            block_set.add(chosen_block)
            
            # Update uncovered
            for sub in itertools.combinations(chosen_block, t):
                sub_id = subset_to_id[sub]
                if uncovered[sub_id]:
                    uncovered[sub_id] = 0
                    sub_idx = id_to_idx[sub_id]
                    last_idx = len(uncovered_ids) - 1
                    temp_id = uncovered_ids[sub_idx]
                    uncovered_ids[sub_idx] = uncovered_ids[last_idx]
                    uncovered_ids[last_idx] = temp_id
                    if sub_idx != last_idx:
                        id_to_idx[uncovered_ids[sub_idx]] = sub_idx
                    uncovered_ids.pop()
                    del id_to_idx[sub_id]
                    
        current_len = len(blocks)
        if current_len < best_len:
            best_len = current_len
            best_blocks = blocks
            
        # Local search phase: try to improve by replacing blocks
        # This is a heuristic to escape local optima
        if best_blocks and time.time() < deadline - 5.0:
            # Try a few local search moves
            for _ in range(10):
                if time.time() > deadline - 2.0:
                    break
                # Pick a random block to remove
                if not best_blocks:
                    break
                rem_idx = random.randrange(len(best_blocks))
                # Check if removing it still covers everything
                # This is expensive, so we'll just try it
                # For now, skip local search to keep it simple and fast
                pass

    return best_blocks
