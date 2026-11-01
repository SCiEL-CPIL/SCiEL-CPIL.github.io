"""Covering design C(19,8,5) - Global Bitmask Greedy with Local Search."""
import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    # 1. Generate all t-subsets and assign indices
    all_t_subs = list(itertools.combinations(range(v), t))
    n_subs = len(all_t_subs)
    
    # Map subset tuple to index
    sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}

    # 2. Generate all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # 3. Precompute bitmask for each block
    # block_mask[b] is an integer where bit i is 1 if block b covers t-subset i
    block_masks = [0] * n_blocks
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        # Generate all t-subsets of this k-block
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_masks[b_idx] = mask

    best_blocks = None
    best_count = float('inf')
    
    rng = random.Random(42)
    max_restarts = 100  # Bitmask ops are fast, allow more restarts
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        uncovered_mask = (1 << n_subs) - 1
        selected_blocks = []
        
        # Optimization: Keep a list of active block indices
        active_blocks = list(range(n_blocks))
        
        while uncovered_mask:
            if time.time() > deadline:
                break
                
            best_gain = 0
            best_block_candidates = []
            
            new_active_blocks = []
            for b_idx in active_blocks:
                mask = block_masks[b_idx]
                gain = (mask & uncovered_mask).bit_count()
                if gain > 0:
                    new_active_blocks.append(b_idx)
                    if gain > best_gain:
                        best_gain = gain
                        best_block_candidates = [b_idx]
                    elif gain == best_gain:
                        best_block_candidates.append(b_idx)
                        
            if not best_block_candidates:
                break
                
            # Randomized tie-breaking
            chosen_block = rng.choice(best_block_candidates)
            selected_blocks.append(chosen_block)
            uncovered_mask &= ~block_masks[chosen_block]
            active_blocks = new_active_blocks
            
            if not active_blocks:
                break

        if uncovered_mask == 0:
            # Local Search: Delete redundant blocks
            # Compute cover counts
            cover_counts = [0] * n_subs
            for b_idx in selected_blocks:
                mask = block_masks[b_idx]
                m = mask
                while m:
                    idx = (m & -m).bit_length() - 1
                    cover_counts[idx] += 1
                    m &= m - 1
            
            changed = True
            while changed:
                changed = False
                for i in range(len(selected_blocks)):
                    b_idx = selected_blocks[i]
                    mask = block_masks[b_idx]
                    m = mask
                    is_redundant = True
                    while m:
                        idx = (m & -m).bit_length() - 1
                        if cover_counts[idx] == 1:
                            is_redundant = False
                            break
                        m &= m - 1
                            
                    if is_redundant:
                        # Remove block i
                        m = mask
                        while m:
                            idx = (m & -m).bit_length() - 1
                            cover_counts[idx] -= 1
                            m &= m - 1
                        selected_blocks.pop(i)
                        changed = True
                        break # Restart loop
            
            if len(selected_blocks) < best_count:
                best_count = len(selected_blocks)
                best_blocks = [list(all_blocks[i]) for i in selected_blocks]
                
                if best_count <= 335:
                    break

    if best_blocks is None:
        # Fallback
        return []
        
    return best_blocks
