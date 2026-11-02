import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    # 1. Precomputation
    t_subsets = list(itertools.combinations(range(v), t))
    num_t = len(t_subsets)
    t_idx = {ts: i for i, ts in enumerate(t_subsets)}
    
    k_blocks = list(itertools.combinations(range(v), k))
    num_k = len(k_blocks)
    
    cover_masks = [0] * num_k
    subset_to_blocks = [[] for _ in range(num_t)]
    
    for b_idx, block in enumerate(k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_idx[ts]
            mask |= (1 << idx)
            subset_to_blocks[idx].append(b_idx)
        cover_masks[b_idx] = mask

    # 2. Greedy Initialization
    uncovered_mask = (1 << num_t) - 1
    current_blocks = []
    in_cover = [False] * num_k
    
    while uncovered_mask:
        best_b = -1
        best_gain = 0
        for b in range(num_k):
            if in_cover[b]:
                continue
            gain = (cover_masks[b] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b = b
        if best_b == -1:
            break
        current_blocks.append(best_b)
        in_cover[best_b] = True
        uncovered_mask &= ~cover_masks[best_b]
        
    def reduce_cover():
        """Removes redundant blocks. Returns list of removed block indices."""
        removed = []
        changed = True
        while changed:
            changed = False
            for i in range(len(current_blocks) - 1, -1, -1):
                b = current_blocks[i]
                if (cover_masks[b] & uncovered_mask) == 0:
                    in_cover[b] = False
                    current_blocks.pop(i)
                    removed.append(b)
                    changed = True
                    break
        return removed

    reduce_cover()
    
    best_size = len(current_blocks)
    best_blocks_copy = list(current_blocks)
    best_in_cover_copy = list(in_cover)
    best_uncovered_copy = uncovered_mask
    
    stall_count = 0
    
    while time.time() - start_time < time_limit:
        improved = False
        
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            if not current_blocks:
                break
                
            i = random.randint(0, len(current_blocks) - 1)
            old_b = current_blocks[i]
            
            common_mask = 0
            for j, b in enumerate(current_blocks):
                if j != i:
                    common_mask |= cover_masks[b]
                    
            unique_mask = cover_masks[old_b] & ~common_mask
            
            if unique_mask == 0:
                in_cover[old_b] = False
                current_blocks.pop(i)
                improved = True
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks_copy = list(current_blocks)
                    best_in_cover_copy = list(in_cover)
                    best_uncovered_copy = uncovered_mask
                break
            
            unique_indices = []
            m = unique_mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                unique_indices.append(idx)
                m ^= lsb
                
            if not unique_indices:
                continue

            min_list_idx = min(unique_indices, key=lambda x: len(subset_to_blocks[x]))
            candidates = set(subset_to_blocks[min_list_idx])
            
            for idx in unique_indices:
                if idx == min_list_idx:
                    continue
                candidates &= set(subset_to_blocks[idx])
                if not candidates:
                    break
                    
            valid_candidates = [b for b in candidates if not in_cover[b]]
            
            if not valid_candidates:
                continue
                
            new_b = random.choice(valid_candidates)
            
            # Save state
            prev_size = len(current_blocks)
            prev_uncovered = uncovered_mask
            prev_in_cover_state = in_cover[new_b] # False
            # We need to save the entire in_cover array? No, just the ones we change.
            # We change old_b (False) and new_b (True).
            # And reduction might remove others.
            
            # Perform swap
            in_cover[old_b] = False
            in_cover[new_b] = True
            current_blocks[i] = new_b
            uncovered_mask &= ~cover_masks[new_b]
            
            # Reduce
            removed_blocks = reduce_cover()
            
            if len(current_blocks) < prev_size:
                improved = True
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks_copy = list(current_blocks)
                    best_in_cover_copy = list(in_cover)
                    best_uncovered_copy = uncovered_mask
                break
            else:
                # Revert
                # Restore removed blocks
                for rb in removed_blocks:
                    in_cover[rb] = True
                    current_blocks.append(rb) # Order doesn't matter for set, but for list it does?
                    # We need to restore them to their original positions? 
                    # No, just having them in the list is enough for the set logic.
                    # But we need to ensure they are not in the list twice.
                    # Since we popped them, adding them back is fine.
                
                # Restore swap
                current_blocks[i] = old_b
                in_cover[new_b] = False
                in_cover[old_b] = True
                uncovered_mask = prev_uncovered
                
        if not improved:
            stall_count += 1
            if stall_count > 10:
                # Kick
                num_remove = min(3, len(current_blocks))
                for _ in range(num_remove):
                    if current_blocks:
                        idx = random.randint(0, len(current_blocks)-1)
                        b = current_blocks.pop(idx)
                        in_cover[b] = False
                        
                covered_mask = 0
                for b in current_blocks:
                    covered_mask |= cover_masks[b]
                uncovered_mask = (1 << num_t) - 1
                uncovered_mask &= ~covered_mask
                
                while uncovered_mask:
                    best_b = -1
                    best_gain = 0
                    for b in range(num_k):
                        if in_cover[b]:
                            continue
                        gain = (cover_masks[b] & uncovered_mask).bit_count()
                        if gain > best_gain:
                            best_gain = gain
                            best_b = b
                    if best_b == -1:
                        break
                    current_blocks.append(best_b)
                    in_cover[best_b] = True
                    uncovered_mask &= ~cover_masks[best_b]
                    
                reduce_cover()
                
                if len(current_blocks) < best_size:
                    best_size = len(current_blocks)
                    best_blocks_copy = list(current_blocks)
                    best_in_cover_copy = list(in_cover)
                    best_uncovered_copy = uncovered_mask
                    stall_count = 0
                else:
                    # Revert to best
                    current_blocks = list(best_blocks_copy)
                    in_cover = list(best_in_cover_copy)
                    uncovered_mask = best_uncovered_copy
                    stall_count = 0
        else:
            stall_count = 0

    current_blocks = list(best_blocks_copy)
    return [list(k_blocks[i]) for i in current_blocks]
