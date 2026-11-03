import itertools
import time
import random
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    points = list(range(v))
    
    # Map each t-subset to an index
    t_subsets = list(itertools.combinations(points, t))
    n_t = len(t_subsets)
    t_subset_index = {ts: i for i, ts in enumerate(t_subsets)}
    
    # Precompute masks for all possible blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    for i, block in enumerate(all_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = t_subset_index[ts]
            mask |= (1 << idx)
        block_masks[i] = mask
    
    # Helper: count coverage
    def get_counts(selected):
        counts = [0] * n_t
        for b_idx in selected:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts
    
    # 1. Greedy Initialization with lookahead
    uncovered_mask = (1 << n_t) - 1
    selected_blocks = []
    
    while uncovered_mask:
        if time.time() > deadline - 30:
            break
        best_block_idx = -1
        best_gain = 0
        # Use a more sophisticated scoring: gain / (1 + something)
        for i in range(n_blocks):
            gain = (uncovered_mask & block_masks[i]).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = i
        if best_block_idx == -1:
            break
        selected_blocks.append(best_block_idx)
        uncovered_mask &= ~block_masks[best_block_idx]
        
    # 2. Aggressive Local Search: Remove Redundant Blocks (1-opt)
    def get_counts(selected):
        counts = [0] * n_t
        for b_idx in selected:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    counts = get_counts(selected_blocks)
    changed = True
    while changed and time.time() < deadline - 20:
        changed = False
        new_selected = []
        for b_idx in selected_blocks:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
        selected_blocks = new_selected

    best_size = len(selected_blocks)
    best_selected = list(selected_blocks)
    
    # 3. Enhanced Simulated Annealing with multiple moves
    current_selected = list(selected_blocks)
    current_counts = get_counts(current_selected)
    current_size = len(current_selected)
    
    T = 1.5
    cooling_rate = 0.98
    min_T = 0.005
    
    all_block_indices = list(range(n_blocks))
    
    while time.time() < deadline - 5:
        if T < min_T:
            break
            
        r = random.random()
        
        if r < 0.25:
            # Try Remove
            if not current_selected: continue
            b_idx = random.choice(current_selected)
            mask = block_masks[b_idx]
            valid = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if current_counts[idx] <= 1:
                    valid = False
                    break
                m ^= lsb
            if valid:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] -= 1
                    m ^= lsb
                current_selected.remove(b_idx)
                new_size = current_size - 1
                current_size = new_size
                if current_size < best_size:
                    best_size = current_size
                    best_selected = list(current_selected)
        elif r < 0.55:
            # Try Add
            b_idx = random.choice(all_block_indices)
            if b_idx in current_selected: continue
            delta = 1
            if random.random() < math.exp(-delta / T):
                mask = block_masks[b_idx]
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] += 1
                    m ^= lsb
                current_selected.append(b_idx)
                current_size += 1
        elif r < 0.80:
            # Try Swap
            if not current_selected: continue
            b_idx = random.choice(current_selected)
            b_idx_new = random.choice(all_block_indices)
            if b_idx == b_idx_new: continue
            if b_idx_new in current_selected: continue
            
            mask_old = block_masks[b_idx]
            mask_new = block_masks[b_idx_new]
            
            critical_subsets = []
            m = mask_old
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if current_counts[idx] == 1:
                    critical_subsets.append(idx)
                m ^= lsb
                
            if not critical_subsets:
                valid = True
            else:
                valid = True
                for idx in critical_subsets:
                    if not (mask_new & (1 << idx)):
                        valid = False
                        break
            
            if valid:
                m = mask_old
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] -= 1
                    m ^= lsb
                m = mask_new
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    current_counts[idx] += 1
                    m ^= lsb
                
                pos = current_selected.index(b_idx)
                current_selected[pos] = b_idx_new
        else:
            # Try 2-swap: remove two, add one (net -1)
            if len(current_selected) < 2: continue
            i, j = random.sample(range(len(current_selected)), 2)
            b_idx1 = current_selected[i]
            b_idx2 = current_selected[j]
            
            mask1 = block_masks[b_idx1]
            mask2 = block_masks[b_idx2]
            
            # Check if removing both is valid (all subsets still covered)
            valid = True
            m = mask1 | mask2
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if current_counts[idx] <= 2:
                    valid = False
                    break
                m ^= lsb
            
            if valid:
                # Try to find a replacement block that covers the critical subsets
                # Critical subsets are those with count exactly 2 that are in mask1|mask2
                critical = []
                m = mask1 | mask2
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if current_counts[idx] == 2:
                        critical.append(idx)
                    m ^= lsb
                
                if not critical:
                    # Just remove both
                    m = mask1
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        current_counts[idx] -= 1
                        m ^= lsb
                    m = mask2
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        current_counts[idx] -= 1
                        m ^= lsb
                    
                    # Remove from list (handle indices carefully)
                    if i > j:
                        i, j = j, i
                    current_selected.pop(i)
                    current_selected.pop(j)
                    current_size -= 2
                    if current_size < best_size:
                        best_size = current_size
                        best_selected = list(current_selected)
                else:
                    # Try to find a block that covers all critical subsets
                    # and is not already in the set
                    candidate_mask = 0
                    for idx in critical:
                        candidate_mask |= (1 << idx)
                    
                    found = False
                    for b_idx_new in random.sample(all_block_indices, min(50, n_blocks)):
                        if b_idx_new in current_selected:
                            continue
                        mask_new = block_masks[b_idx_new]
                        if (mask_new & candidate_mask) == candidate_mask:
                            # This block covers all critical subsets
                            # Remove b_idx1 and b_idx2, add b_idx_new
                            m = mask1
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                current_counts[idx] -= 1
                                m ^= lsb
                            m = mask2
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                current_counts[idx] -= 1
                                m ^= lsb
                            m = mask_new
                            while m:
                                lsb = m & -m
                                idx = lsb.bit_length() - 1
                                current_counts[idx] += 1
                                m ^= lsb
                            
                            if i > j:
                                i, j = j, i
                            current_selected.pop(i)
                            current_selected.pop(j)
                            current_selected.append(b_idx_new)
                            current_size -= 1
                            if current_size < best_size:
                                best_size = current_size
                                best_selected = list(current_selected)
                            found = True
                            break
    
    # Final cleanup: remove redundant blocks
    best_selected = list(best_selected)
    counts = get_counts(best_selected)
    changed = True
    while changed and time.time() < deadline - 2:
        changed = False
        new_selected = []
        for b_idx in best_selected:
            mask = block_masks[b_idx]
            redundant = True
            m = mask
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                if counts[idx] <= 1:
                    redundant = False
                    break
                m ^= lsb
            if redundant:
                m = mask
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                changed = True
            else:
                new_selected.append(b_idx)
        best_selected = new_selected

    return [list(all_blocks[i]) for i in best_selected]
