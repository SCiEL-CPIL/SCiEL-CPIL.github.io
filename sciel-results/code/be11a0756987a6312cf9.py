import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0
    
    all_points = list(range(v))
    all_t_subsets = list(itertools.combinations(all_points, t))
    num_t_subsets = len(all_t_subsets)
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Greedy Initialization
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    while uncovered:
        if time.time() > deadline - 20: 
            break # Safety, though greedy is fast
        
        # Pick a random uncovered triple to add diversity
        target_idx = random.choice(list(uncovered))
        target_ts = all_t_subsets[target_idx]
        
        best_block = None
        best_gain = -1
        
        remaining_points = [p for p in all_points if p not in target_ts]
        
        for combo in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(target_ts + combo))
            
            gain = 0
            for ts in itertools.combinations(block, t):
                idx = t_subset_to_idx[ts]
                if idx in uncovered:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
            
        blocks.append(best_block)
        
        for ts in itertools.combinations(best_block, t):
            idx = t_subset_to_idx[ts]
            if idx in uncovered:
                uncovered.discard(idx)
                
    if not uncovered:
        # Valid covering found
        pass
    else:
        # Fallback: If greedy didn't finish, we have an invalid covering.
        # We must finish it.
        while uncovered:
             target_idx = random.choice(list(uncovered))
             target_ts = all_t_subsets[target_idx]
             best_block = None
             best_gain = -1
             remaining_points = [p for p in all_points if p not in target_ts]
             for combo in itertools.combinations(remaining_points, k - t):
                 block = tuple(sorted(target_ts + combo))
                 gain = 0
                 for ts in itertools.combinations(block, t):
                     idx = t_subset_to_idx[ts]
                     if idx in uncovered:
                         gain += 1
                 if gain > best_gain:
                     best_gain = gain
                     best_block = block
             blocks.append(best_block)
             for ts in itertools.combinations(best_block, t):
                 idx = t_subset_to_idx[ts]
                 if idx in uncovered:
                     uncovered.discard(idx)

    # Local Search
    def count_covers(blocks):
        counts = [0] * num_t_subsets
        for b in blocks:
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                counts[idx] += 1
        return counts

    counts = count_covers(blocks)
    
    def try_remove_redundant(blocks, counts):
        removed = False
        new_blocks = []
        for b in blocks:
            redundant = True
            for ts in itertools.combinations(b, t):
                idx = t_subset_to_idx[ts]
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                for ts in itertools.combinations(b, t):
                    idx = t_subset_to_idx[ts]
                    counts[idx] -= 1
                removed = True
            else:
                new_blocks.append(b)
        return new_blocks, counts, removed

    blocks, counts, _ = try_remove_redundant(blocks, counts)
    
    best_blocks = list(blocks)
    best_len = len(blocks)
    
    while time.time() < deadline:
        improved = False
        
        # 1. Remove Redundant
        new_blocks, new_counts, removed = try_remove_redundant(blocks, counts)
        if removed:
            blocks = new_blocks
            counts = new_counts
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = list(blocks)
            improved = True
        
        # 2. Random Swap
        if len(blocks) > 0:
            idx_to_swap = random.randint(0, len(blocks) - 1)
            old_block = blocks[idx_to_swap]
            new_block = tuple(sorted(random.sample(all_points, k)))
            
            valid_swap = True
            new_block_set = set(itertools.combinations(new_block, t))
            
            for ts in itertools.combinations(old_block, t):
                idx = t_subset_to_idx[ts]
                if counts[idx] == 1 and ts not in new_block_set:
                    valid_swap = False
                    break
            
            if valid_swap:
                for ts in itertools.combinations(old_block, t):
                    counts[t_subset_to_idx[ts]] -= 1
                for ts in itertools.combinations(new_block, t):
                    counts[t_subset_to_idx[ts]] += 1
                
                blocks[idx_to_swap] = new_block
                
                new_blocks, new_counts, removed = try_remove_redundant(blocks, counts)
                
                if removed:
                    blocks = new_blocks
                    counts = new_counts
                    if len(blocks) < best_len:
                        best_len = len(blocks)
                        best_blocks = list(blocks)
                    improved = True
                else:
                    # Revert
                    for ts in itertools.combinations(new_block, t):
                        counts[t_subset_to_idx[ts]] -= 1
                    for ts in itertools.combinations(old_block, t):
                        counts[t_subset_to_idx[ts]] += 1
                    blocks[idx_to_swap] = old_block
        
        # 3. Add Block then Remove
        if random.random() < 0.2:
            new_block = tuple(sorted(random.sample(all_points, k)))
            # Add it
            blocks.append(new_block)
            for ts in itertools.combinations(new_block, t):
                counts[t_subset_to_idx[ts]] += 1
            
            new_blocks, new_counts, removed = try_remove_redundant(blocks, counts)
            
            if len(new_blocks) < len(blocks):
                blocks = new_blocks
                counts = new_counts
                if len(blocks) < best_len:
                    best_len = len(blocks)
                    best_blocks = list(blocks)
                improved = True
            else:
                # Revert
                blocks.pop()
                for ts in itertools.combinations(new_block, t):
                    counts[t_subset_to_idx[ts]] -= 1

    return best_blocks
