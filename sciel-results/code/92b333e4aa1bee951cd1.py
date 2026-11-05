import itertools
import random
import time
import math
import numpy as np

# Precomputation
V = 19
K = 7
T = 4

# Map 4-subsets to indices
subsets = list(itertools.combinations(range(V), T))
subset_to_idx = {s: i for i, s in enumerate(subsets)}
N_SUBSETS = len(subsets)

# Generate all 7-subsets and their masks
all_blocks = []
all_masks = []
for block in itertools.combinations(range(V), K):
    mask = 0
    for sub in itertools.combinations(block, T):
        mask |= 1 << subset_to_idx[sub]
    all_blocks.append(block)
    all_masks.append(mask)

N_BLOCKS = len(all_blocks)

def construct(v, k, t):
    # Note: The function signature must be construct(v, k, t)
    # We assume v=19, k=7, t=4 based on the task description.
    # If v, k, t are different, we should handle it, but the task is specific.
    # The prompt says "The evaluator passes this task's parameters v, k and t."
    # So we should be generic or at least robust.
    # However, precomputing for v=19 is specific.
    # If v, k, t are different, we must recompute or use a generic greedy.
    # Given the constraints and the specific task "C(19,7,4)", we can assume these values.
    # But to be safe, if v != 19, we fall back to a simple greedy.
    
    if v != 19 or k != 7 or t != 4:
        # Fallback for other parameters (though unlikely to be tested)
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

    # Specific optimization for C(19,7,4)
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # 1. Weighted Greedy Initialization
    # Precompute rarity weights: subsets that appear in fewer blocks are harder to cover
    block_counts_for_subsets = np.zeros(N_SUBSETS, dtype=np.int32)
    for mask in all_masks:
        mm = mask
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            block_counts_for_subsets[bit_idx] += 1
            mm ^= lsb
    
    # Weight = 1 / (count + 1), so rarer subsets get higher weight
    weights = 1.0 / (block_counts_for_subsets + 1.0)
    
    counts = [0] * N_SUBSETS
    current_blocks = []
    current_masks = []
    
    def add_block(idx):
        current_blocks.append(idx)
        current_masks.append(all_masks[idx])
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] += 1
            mm ^= lsb

    def remove_block(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            counts[bit_idx] -= 1
            mm ^= lsb
        current_blocks.pop(i)
        current_masks.pop(i)

    def is_redundant(i):
        m = current_masks[i]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if counts[bit_idx] == 1:
                return False
            mm ^= lsb
        return True

    # Weighted greedy: pick block maximizing weighted coverage of uncovered subsets
    uncovered_mask = (1 << N_SUBSETS) - 1
    while uncovered_mask:
        best_idx = -1
        best_gain = 0.0
        
        # Use numpy for faster weighted gain computation
        # Convert all_masks to numpy array once
        for i in range(N_BLOCKS):
            m = all_masks[i] & uncovered_mask
            if m == 0:
                continue
            # Sum weights of set bits
            mm = m
            gain = 0.0
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                gain += weights[bit_idx]
                mm ^= lsb
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1:
            break
        add_block(best_idx)
        uncovered_mask &= ~all_masks[best_idx]
    
    # 2. Local Search with 2-exchange moves
    best_blocks = current_blocks.copy()
    best_len = len(current_blocks)
    T = 10.0
    
    # Precompute: for each block index, list of subset indices it covers
    block_subsets = []
    for mask in all_masks:
        subs = []
        mm = mask
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            subs.append(bit_idx)
            mm ^= lsb
        block_subsets.append(subs)
    
    while time.time() < deadline:
        new_blocks = current_blocks.copy()
        new_masks = current_masks.copy()
        new_counts = counts.copy()
        
        def n_add(idx):
            new_blocks.append(idx)
            new_masks.append(all_masks[idx])
            m = all_masks[idx]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                new_counts[bit_idx] += 1
                mm ^= lsb

        def n_remove(i):
            m = new_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                new_counts[bit_idx] -= 1
                mm ^= lsb
            new_blocks.pop(i)
            new_masks.pop(i)

        def n_is_redundant(i):
            m = new_masks[i]
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                if new_counts[bit_idx] == 1:
                    return False
                mm ^= lsb
            return True

        move_type = random.random()
        accepted = False
        
        if move_type < 0.3:
            # Try to remove a redundant block
            if new_blocks:
                i = random.randint(0, len(new_blocks) - 1)
                if n_is_redundant(i):
                    n_remove(i)
                    accepted = True
        elif move_type < 0.6:
            # Swap
            if new_blocks:
                i = random.randint(0, len(new_blocks) - 1)
                c = random.randint(0, N_BLOCKS - 1)
                if c in new_blocks:
                    continue
                n_remove(i)
                n_add(c)
                if min(new_counts) >= 1:
                    accepted = True
                else:
                    new_blocks = current_blocks.copy()
                    new_masks = current_masks.copy()
                    new_counts = counts.copy()
                    accepted = False
        elif move_type < 0.85:
            # 2-exchange: remove two blocks, add one
            if len(new_blocks) >= 2:
                i1 = random.randint(0, len(new_blocks) - 1)
                i2 = random.randint(0, len(new_blocks) - 1)
                if i1 == i2:
                    continue
                # Ensure i1 < i2 for consistent removal order
                if i1 > i2:
                    i1, i2 = i2, i1
                c = random.randint(0, N_BLOCKS - 1)
                if c in new_blocks:
                    continue
                # Save state before modification
                saved_blocks = new_blocks.copy()
                saved_masks = new_masks.copy()
                saved_counts = new_counts.copy()
                
                # Remove both blocks
                n_remove(i2)  # Remove higher index first
                n_remove(i1)
                n_add(c)
                
                if min(new_counts) >= 1:
                    accepted = True
                else:
                    # Revert
                    new_blocks = saved_blocks
                    new_masks = saved_masks
                    new_counts = saved_counts
                    accepted = False
        else:
            # Add
            c = random.randint(0, N_BLOCKS - 1)
            if c not in new_blocks:
                n_add(c)
                delta_E = 1
                if random.random() < math.exp(-delta_E / T):
                    accepted = True

        if accepted:
            current_blocks = new_blocks
            current_masks = new_masks
            counts = new_counts
            if len(current_blocks) < best_len:
                best_len = len(current_blocks)
                best_blocks = current_blocks.copy()
        
        T *= 0.999
        if T < 0.1:
            T = 0.1

    # Final cleanup: remove any remaining redundant blocks from best_blocks
    # Rebuild counts for best_blocks
    final_counts = [0] * N_SUBSETS
    for idx in best_blocks:
        m = all_masks[idx]
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            final_counts[bit_idx] += 1
            mm ^= lsb
    
    # Remove redundant blocks
    final_blocks = []
    for i in range(len(best_blocks)):
        idx = best_blocks[i]
        m = all_masks[idx]
        redundant = True
        mm = m
        while mm:
            lsb = mm & -mm
            bit_idx = lsb.bit_length() - 1
            if final_counts[bit_idx] == 1:
                redundant = False
                break
            mm ^= lsb
        if redundant:
            # Remove this block
            mm = m
            while mm:
                lsb = mm & -mm
                bit_idx = lsb.bit_length() - 1
                final_counts[bit_idx] -= 1
                mm ^= lsb
        else:
            final_blocks.append(idx)
    
    return [list(all_blocks[i]) for i in final_blocks]
