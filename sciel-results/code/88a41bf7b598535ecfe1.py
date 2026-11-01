import numpy as np
import time
import itertools

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 148  # safety margin before 150s
    
    # Precompute all t-subsets and their indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_to_idx = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-blocks
    all_kblocks = list(itertools.combinations(range(v), k))
    num_kblocks = len(all_kblocks)
    
    # For each block, store the indices of t-subsets it covers
    num_t_per_block = len(list(itertools.combinations(range(k), t)))
    
    block_tsub_indices = np.empty((num_kblocks, num_t_per_block), dtype=np.int32)
    for i, block in enumerate(all_kblocks):
        for j, tsub in enumerate(itertools.combinations(block, t)):
            block_tsub_indices[i, j] = tsub_to_idx[tsub]
    
    # Greedy set cover
    uncovered = np.ones(num_tsubs, dtype=bool)
    chosen_indices = []
    used_block_indices = set()
    
    while True:
        if time.time() > deadline - 40:
            break
        if not uncovered.any():
            break
        
        # Compute gains for all unused blocks
        gains = uncovered[block_tsub_indices].sum(axis=1)
        
        # Mask out already-used blocks
        for bi in used_block_indices:
            gains[bi] = -1
        
        best_block_idx = int(np.argmax(gains))
        if gains[best_block_idx] <= 0:
            break
        
        # Add this block
        chosen_indices.append(best_block_idx)
        used_block_indices.add(best_block_idx)
        
        # Update uncovered
        uncovered[block_tsub_indices[best_block_idx]] = False
    
    # If greedy didn't finish, do a faster greedy to fill remaining
    if uncovered.any() and time.time() < deadline - 10:
        while uncovered.any():
            gains = uncovered[block_tsub_indices].sum(axis=1)
            for bi in used_block_indices:
                gains[bi] = -1
            best_block_idx = int(np.argmax(gains))
            if gains[best_block_idx] <= 0:
                break
            chosen_indices.append(best_block_idx)
            used_block_indices.add(best_block_idx)
            uncovered[block_tsub_indices[best_block_idx]] = False
    
    # Local search: redundancy removal
    if time.time() < deadline - 15:
        # Build coverage count array
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for bi in chosen_indices:
            cover_count[block_tsub_indices[bi]] += 1
        
        # Try removing blocks that don't uniquely cover any t-subset
        improved = True
        max_passes = 5
        for p in range(max_passes):
            if time.time() > deadline - 10:
                break
            improved = False
            new_chosen = []
            for bi in chosen_indices:
                # Check if this block uniquely covers any t-subset
                # A t-subset is uniquely covered if cover_count[ts] == 1 and this block covers it
                # We need to check: for each t-subset in this block, is cover_count == 1?
                block_tsubs = block_tsub_indices[bi]
                # If any t-subset in this block has cover_count == 1, we can't remove it
                if np.any(cover_count[block_tsubs] == 1):
                    new_chosen.append(bi)
                else:
                    # Remove this block
                    cover_count[block_tsubs] -= 1
                    improved = True
            chosen_indices = new_chosen
            if not improved:
                break
    
    # Local search: swap-based improvement
    if time.time() < deadline - 10:
        # For each chosen block, try to find a replacement that covers more uncovered
        # First, compute what's covered by the current set
        covered = np.zeros(num_tsubs, dtype=np.int32)
        for bi in chosen_indices:
            covered[block_tsub_indices[bi]] += 1
        
        # Try swapping each block with a better one
        for _ in range(3):  # multiple passes
            if time.time() > deadline - 5:
                break
            any_swap = False
            for idx, bi in enumerate(list(chosen_indices)):
                if time.time() > deadline - 5:
                    break
                # What t-subsets would become uncovered if we remove this block?
                would_uncover = (covered[block_tsub_indices[bi]] == 1)
                num_would_uncover = int(np.sum(would_uncover))
                if num_would_uncover == 0:
                    # Can remove without replacement
                    chosen_indices.pop(idx)
                    covered[block_tsub_indices[bi]] -= 1
                    any_swap = True
                    idx -= 1  # adjust index after pop
                    continue
                
                # Find best replacement: a block not in chosen that covers all would_uncover t-subsets
                # and maximizes additional coverage
                # This is expensive; do a limited search
                # Compute which blocks cover all the would_uncover t-subsets
                # For efficiency, just check blocks that cover at least one would_uncover t-subset
                # and count how many they cover
                
                # Get the set of t-subsets that would be uncovered
                target_tsubs = block_tsub_indices[bi][would_uncover]
                
                # Find candidate blocks: those that cover all target_tsubs
                # For each candidate, count total new coverage
                # This is O(num_kblocks * C(k,t)) which might be too slow
                # Instead, do a limited random sample
                num_candidates = min(500, num_kblocks)
                if num_kblocks > num_candidates:
                    candidates = np.random.choice(num_kblocks, num_candidates, replace=False)
                else:
                    candidates = np.arange(num_kblocks)
                
                best_candidate = -1
                best_score = 0
                for ci in candidates:
                    if ci in chosen_indices:
                        continue
                    cand_tsubs = block_tsub_indices[ci]
                    # Check if candidate covers all target_tsubs
                    # target_tsubs are specific indices; check if all appear in cand_tsubs
                    cand_set = set(cand_tsubs.tolist())
                    if all(ts in cand_set for ts in target_tsubs.tolist()):
                        # Count additional coverage
                        additional = int(np.sum(covered[cand_tsubs] == 0))
                        if additional > best_score:
                            best_score = additional
                            best_candidate = ci
                
                if best_candidate >= 0:
                    # Perform swap
                    covered[block_tsub_indices[bi]] -= 1
                    covered[block_tsub_indices[best_candidate]] += 1
                    chosen_indices[idx] = best_candidate
                    any_swap = True
            
            if not any_swap:
                break
    
    # Final validation
    result = [list(all_kblocks[bi]) for bi in chosen_indices]
    
    # Verify all t-subsets are covered
    final_uncovered = np.zeros(num_tsubs, dtype=bool)
    for bi in chosen_indices:
        final_uncovered[block_tsub_indices[bi]] = True
    
    if not final_uncovered.all():
        # Fallback: greedy to fill gaps
        uncovered_mask = ~final_uncovered
        while uncovered_mask.any() and time.time() < deadline - 1:
            gains = uncovered_mask[block_tsub_indices].sum(axis=1)
            best_block_idx = int(np.argmax(gains))
            if gains[best_block_idx] <= 0:
                break
            chosen_indices.append(best_block_idx)
            final_uncovered[block_tsub_indices[best_block_idx]] = True
            uncovered_mask = ~final_uncovered
    
    result = [list(all_kblocks[bi]) for bi in chosen_indices]
    return result
