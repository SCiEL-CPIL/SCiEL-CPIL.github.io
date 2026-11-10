import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin for 150s limit
    
    # Total number of t-subsets
    total_t_subsets = 1
    for i in range(t):
        total_t_subsets *= (v - i)
        total_t_subsets //= (i + 1)
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    
    # Track coverage: for each t-subset, how many blocks cover it
    # We'll use a dictionary: t_subset -> list of block indices that cover it
    # But for efficiency, let's use a count array and a coverage structure
    
    # Better approach: maintain for each t-subset a count of covering blocks
    # And for each block, the set of t-subsets it covers
    
    # Greedy initialization
    uncovered_count = total_t_subsets
    # Map each t-subset to an index for efficient tracking
    t_subset_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # coverage[i] = number of blocks covering t-subset i
    coverage = [0] * total_t_subsets
    
    # For each block, store the t-subsets it covers
    block_t_subsets = []  # list of lists of t-subset indices
    
    # Greedy: repeatedly pick the best block to add
    # A block's "gain" = number of currently uncovered t-subsets it covers
    # To speed up, we can use a heuristic: start with random blocks and improve
    
    # Actually, let's do a smarter greedy:
    # 1. Start with no blocks
    # 2. At each step, find the block (k-subset) that covers the most uncovered t-subsets
    #    This is NP-hard, so use a heuristic:
    #    - Pick a random uncovered t-subset
    #    - Extend it to a k-subset by adding points that maximize coverage of other uncovered t-subsets
    
    def greedy_construct():
        nonlocal coverage, block_t_subsets
        coverage = [0] * total_t_subsets
        block_t_subsets = []
        blocks = []
        
        # Track which t-subsets are covered
        # uncovered[t_subset_idx] = True if not covered
        uncovered = [True] * total_t_subsets
        num_uncovered = total_t_subsets
        
        while num_uncovered > 0:
            if time.time() > deadline - 5:
                break
            
            # Pick a random uncovered t-subset to start
            # Find an uncovered one
            idx = random.randrange(total_t_subsets)
            while not uncovered[idx]:
                idx = (idx + 1) % total_t_subsets
            
            seed_t = all_t_subsets[idx]
            block = list(seed_t)
            
            # Extend to k elements
            while len(block) < k:
                # For each candidate point, compute how many uncovered t-subsets it would cover
                # A t-subset {a,b,c,d} is covered by block if all 4 are in block
                # When we add point x to block, the new t-subsets covered are those
                # that contain x and t-1 elements from the current block
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block:
                        continue
                    gain = 0
                    # t-subsets that include x and t-1 from block
                    for rest in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        ts_idx = t_subset_to_idx[ts]
                        if uncovered[ts_idx]:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    break
                block.append(best_x)
            
            block.sort()
            
            # Add this block
            block_idx = len(blocks)
            blocks.append(block)
            
            # Update coverage
            new_t_subsets = []
            for ts in itertools.combinations(block, t):
                ts_idx = t_subset_to_idx[ts]
                new_t_subsets.append(ts_idx)
                if coverage[ts_idx] == 0:
                    num_uncovered -= 1
                    uncovered[ts_idx] = False
                coverage[ts_idx] += 1
            
            block_t_subsets.append(new_t_subsets)
        
        return blocks
    
    # Run greedy to get initial solution
    blocks = greedy_construct()
    
    # Now try to improve by removing redundant blocks
    def try_remove_block(blocks, coverage, block_t_subsets):
        """Try to remove blocks that are redundant."""
        improved = True
        while improved:
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                # Check if all t-subsets in block i are covered by other blocks
                redundant = True
                for ts_idx in block_t_subsets[i]:
                    if coverage[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for ts_idx in block_t_subsets[i]:
                        coverage[ts_idx] -= 1
                    blocks.pop(i)
                    block_t_subsets.pop(i)
                    improved = True
        return blocks
    
    blocks = try_remove_block(blocks, coverage, block_t_subsets)
    
    # Now try local search: swap blocks to potentially allow removal of others
    # Strategy: for each block, try replacing it with a different block that covers
    # the same unique t-subsets (those with coverage=1)
    
    # Precompute for each block which t-subsets are unique to it (coverage=1)
    def get_unique_t_subsets(block_idx):
        unique = []
        for ts_idx in block_t_subsets[block_idx]:
            if coverage[ts_idx] == 1:
                unique.append(ts_idx)
        return unique
    
    # Local search: try to find a smaller covering
    best_blocks = blocks[:]
    best_count = len(blocks)
    
    # Simulated annealing style: try to reduce block count
    # Move: pick a block, try to replace it with a block that covers its unique t-subsets
    # and also covers some t-subsets that were only covered by another block,
    # allowing that other block to be removed
    
    # Simpler approach: iteratively try to remove blocks by finding alternative coverings
    # for their unique t-subsets
    
    def local_search_improve(blocks, coverage, block_t_subsets, max_iter=1000):
        nonlocal best_blocks, best_count
        current_blocks = blocks[:]
        current_coverage = coverage[:]
        current_block_ts = block_t_subsets[:]
        current_count = len(current_blocks)
        
        for iteration in range(max_iter):
            if time.time() > deadline - 15:
                break
            if current_count <= best_count:
                break
            
            # Try to find a block that can be removed
            # A block can be removed if all its t-subsets are covered by others
            # To enable removal, we might need to modify other blocks
            
            # Pick a random block to try to eliminate
            if not current_blocks:
                break
            target_idx = random.randrange(len(current_blocks))
            target_unique = []
            for ts_idx in current_block_ts[target_idx]:
                if current_coverage[ts_idx] == 1:
                    target_unique.append(ts_idx)
            
            if not target_unique:
                # Block is redundant, remove it
                for ts_idx in current_block_ts[target_idx]:
                    current_coverage[ts_idx] -= 1
                current_blocks.pop(target_idx)
                current_block_ts.pop(target_idx)
                current_count -= 1
                if current_count < best_count:
                    best_count = current_count
                    best_blocks = current_blocks[:]
                continue
            
            # Try to find a replacement block that covers target_unique
            # and allows removal of target
            # Heuristic: start from a random uncovered t-subset in target_unique
            # and build a block that covers it and as many other target_unique as possible
            
            # Pick a random t-subset from target_unique to seed the new block
            seed_idx = random.choice(target_unique)
            seed_t = all_t_subsets[seed_idx]
            new_block = list(seed_t)
            
            # Extend to cover as many target_unique as possible
            remaining_unique = set(target_unique) - {seed_idx}
            
            while len(new_block) < k and remaining_unique:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in new_block:
                        continue
                    gain = 0
                    for rest in itertools.combinations(new_block, t - 1):
                        ts = tuple(sorted(rest + (x,)))
                        try:
                            ts_idx = t_subset_to_idx[ts]
                        except KeyError:
                            continue
                        if ts_idx in remaining_unique:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1 or best_gain == 0:
                    break
                new_block.append(best_x)
                # Update remaining_unique
                for ts in itertools.combinations(new_block, t):
                    try:
                        ts_idx = t_subset_to_idx[ts]
                    except KeyError:
                        continue
                    if ts_idx in remaining_unique:
                        remaining_unique.discard(ts_idx)
            
            new_block.sort()
            
            # Check if new_block covers all target_unique
            new_block_ts = set()
            for ts in itertools.combinations(new_block, t):
                try:
                    new_block_ts.add(t_subset_to_idx[ts])
                except KeyError:
                    pass
            
            if not set(target_unique).issubset(new_block_ts):
                continue
            
            # Now check: if we replace target with new_block, can we remove target?
            # And does new_block allow removal of any other block?
            
            # Simulate the replacement
            # First, what happens to coverage if we remove target?
            # Then add new_block
            
            # Check if new_block is already in current_blocks
            new_block_tuple = tuple(new_block)
            already_exists = any(tuple(b) == new_block_tuple for b in current_blocks)
            
            # Compute the net change in block count:
            # -1 for removing target
            # +1 for adding new_block (if not already exists)
            # Possibly -1 if new_block allows removal of another block
            
            # For simplicity, just do the swap and see if we can then remove target
            # Actually, we're replacing target with new_block, so block count stays same
            # unless new_block makes another block redundant
            
            # Let's just do the swap and then try to remove redundant blocks
            old_block = current_blocks[target_idx]
            old_ts = current_block_ts[target_idx]
            
            # Remove target
            for ts_idx in old_ts:
                current_coverage[ts_idx] -= 1
            current_blocks[target_idx] = new_block
            current_block_ts[target_idx] = list(new_block_ts)
            
            # Add new_block
            for ts_idx in new_block_ts:
                current_coverage[ts_idx] += 1
            
            # Now try to remove redundant blocks
            removed_any = False
            for i in range(len(current_blocks) - 1, -1, -1):
                if i == target_idx:
                    continue
                redundant = True
                for ts_idx in current_block_ts[i]:
                    if current_coverage[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in current_block_ts[i]:
                        current_coverage[ts_idx] -= 1
                    current_blocks.pop(i)
                    current_block_ts.pop(i)
                    current_count -= 1
                    removed_any = True
                    if current_count < best_count:
                        best_count = current_count
                        best_blocks = current_blocks[:]
                    break  # Only remove one at a time
            
            if not removed_any:
                # No improvement, revert
                current_blocks[target_idx] = old_block
                current_block_ts[target_idx] = old_ts
                for ts_idx in new_block_ts:
                    current_coverage[ts_idx] -= 1
                for ts_idx in old_ts:
                    current_coverage[ts_idx] += 1
        
        return best_blocks
    
    # Run local search
    blocks = local_search_improve(blocks, coverage, block_t_subsets)
    
    # Final cleanup: remove any redundant blocks
    def final_cleanup(blocks, coverage, block_t_subsets):
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(blocks) - 1, -1, -1):
                redundant = True
                for ts_idx in block_t_subsets[i]:
                    if coverage[ts_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for ts_idx in block_t_subsets[i]:
                        coverage[ts_idx] -= 1
                    blocks.pop(i)
                    block_t_subsets.pop(i)
                    improved = True
        return blocks
    
    blocks = final_cleanup(blocks, coverage, block_t_subsets)
    
    return blocks
