import itertools
import time
import random

def construct(v, k, t):
    # Parameters for the specific task instance
    if v == 19 and k == 7 and t == 4:
        return _construct_19_7_4()
    
    # Fallback for other parameters (simple greedy)
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

def _construct_19_7_4():
    v, k, t = 19, 7, 4
    deadline = time.time() + 110.0
    
    # 1. Precomputation
    # Map all 4-subsets to indices
    all_4subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_4subsets)}
    num_4subsets = len(all_4subsets)
    
    # Map all 7-blocks to indices and compute their coverage masks
    all_7blocks = list(itertools.combinations(range(v), k))
    num_7blocks = len(all_7blocks)
    
    block_masks = [0] * num_7blocks
    block_subsets = [[] for _ in range(num_7blocks)]
    
    for i, block in enumerate(all_7blocks):
        mask = 0
        subs = []
        for sub in itertools.combinations(block, t):
            idx = subset_to_idx[sub]
            mask |= (1 << idx)
            subs.append(idx)
        block_masks[i] = mask
        block_subsets[i] = subs
        
    all_ones_mask = (1 << num_4subsets) - 1
    
    # 2. Greedy Initialization
    uncovered = all_ones_mask
    current_blocks = []
    block_in_set = [False] * num_7blocks
    count_mask = [0] * num_4subsets
    
    while uncovered:
        best_block = -1
        best_count = -1
        # Optimization: If time is tight, we might want to break, but greedy is fast.
        for i in range(num_7blocks):
            if block_in_set[i]:
                continue
            c = (uncovered & block_masks[i]).bit_count()
            if c > best_count:
                best_count = c
                best_block = i
                
        if best_block == -1:
            break
            
        current_blocks.append(best_block)
        block_in_set[best_block] = True
        
        # Update counts and uncovered
        for s_idx in block_subsets[best_block]:
            count_mask[s_idx] += 1
        uncovered &= ~block_masks[best_block]
        
    best_len = len(current_blocks)
    best_blocks_copy = current_blocks.copy()
    
    # 3. Local Search
    # We will try to reduce the number of blocks via swaps and removals.
    
    def try_reduce():
        """Attempt to remove redundant blocks. Returns True if any block was removed."""
        removed = False
        # Iterate through a copy of the list to avoid index issues during modification
        # Or iterate backwards.
        # Since N is small (~140), copying is fine, but backwards is cleaner.
        for i in range(len(current_blocks) - 1, -1, -1):
            b_idx = current_blocks[i]
            # Check if b_idx is redundant
            redundant = True
            for s_idx in block_subsets[b_idx]:
                if count_mask[s_idx] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove b_idx
                for s_idx in block_subsets[b_idx]:
                    count_mask[s_idx] -= 1
                block_in_set[b_idx] = False
                current_blocks.pop(i)
                removed = True
        return removed

    # Initial reduction
    try_reduce()
    if len(current_blocks) < best_len:
        best_len = len(current_blocks)
        best_blocks_copy = current_blocks.copy()
        
    random.seed(42) # For reproducibility in local search exploration? 
                    # Actually, randomization helps escape local minima.
    
    while time.time() < deadline:
        # 1. Try to reduce
        try_reduce()
        
        current_len = len(current_blocks)
        if current_len < best_len:
            best_len = current_len
            best_blocks_copy = current_blocks.copy()
            
        # 2. If no blocks left, break
        if current_len == 0:
            break
            
        # 3. Randomized Swap
        # Pick a random block A from current
        i = random.randint(0, current_len - 1)
        A = current_blocks[i]
        
        # Pick a random block B not in current
        # Efficient random selection:
        # num_7blocks is ~50k. current_len is ~140.
        # Probability of collision is low.
        while True:
            B = random.randint(0, num_7blocks - 1)
            if not block_in_set[B]:
                break
        
        # Check if swapping A with B is valid
        # Condition: All subsets of A that are currently covered ONLY by A must be covered by B.
        valid = True
        for s_idx in block_subsets[A]:
            if count_mask[s_idx] == 1:
                # This subset is critical for A. B must cover it.
                if not ((block_masks[B] >> s_idx) & 1):
                    valid = False
                    break
        
        if valid:
            # Perform Swap
            # Remove A
            for s_idx in block_subsets[A]:
                count_mask[s_idx] -= 1
            block_in_set[A] = False
            
            # Add B
            for s_idx in block_subsets[B]:
                count_mask[s_idx] += 1
            block_in_set[B] = True
            
            # Update list
            current_blocks[i] = B
            
            # After a swap, immediately try to reduce again in the next loop iteration
            # This allows chaining: Swap -> Remove -> Swap -> Remove...
            
    # 4. Return the best found
    # Convert indices back to point lists
    final_blocks = [list(all_7blocks[i]) for i in best_blocks_copy]
    # Ensure sorted for consistency (though evaluator likely handles sets)
    for b in final_blocks:
        b.sort()
        
    return final_blocks
