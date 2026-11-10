import itertools
import time
import random
import numpy as np

DEADLINE_SECONDS = 115.0

def _generic_greedy(v, k, t, start_time=None):
    if start_time is None:
        start_time = time.time()
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    block_masks = [0] * n_blocks
    blocks_containing = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        for sub in itertools.combinations(block, t):
            s_idx = idx_map[tuple(sub)]
            mask |= (1 << s_idx)
            blocks_containing[s_idx].append(b_idx)
        block_masks[b_idx] = mask

    uncovered = (1 << n_t) - 1
    current_blocks = []
    
    while uncovered and (time.time() - start_time) < DEADLINE_SECONDS - 5:
        # Pick a random uncovered subset
        found_idx = -1
        attempts = 0
        while found_idx == -1 and attempts < 50:
            idx = random.randint(0, n_t - 1)
            if (uncovered >> idx) & 1:
                found_idx = idx
            attempts += 1
        
        if found_idx == -1:
            for idx in range(n_t):
                if (uncovered >> idx) & 1:
                    found_idx = idx
                    break
        
        if found_idx == -1:
            break

        candidates = blocks_containing[found_idx]
        best_gain = -1
        best_block_idx = -1
        
        for b_idx in candidates:
            gain = (block_masks[b_idx] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx != -1:
            current_blocks.append(best_block_idx)
            uncovered &= ~block_masks[best_block_idx]

    return [list(all_blocks[i]) for i in current_blocks]

def construct(v, k, t):
    start_time = time.time()
    
    if (v, k, t) != (19, 8, 5):
        return _generic_greedy(v, k, t, start_time)

    # Precomputation
    points = range(v)
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    idx_map = {s: i for i, s in enumerate(all_t_subsets)}
    
    # Full mask for all t-subsets
    full_mask = (1 << n_t) - 1
    
    # Generate all k-blocks
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute masks for each block and subsets
    block_masks = [0] * n_blocks
    block_subsets = [[] for _ in range(n_blocks)]
    blocks_containing = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(all_blocks):
        mask = 0
        subs = []
        for sub in itertools.combinations(block, t):
            s_idx = idx_map[tuple(sub)]
            mask |= (1 << s_idx)
            subs.append(s_idx)
            blocks_containing[s_idx].append(b_idx)
        block_masks[b_idx] = mask
        block_subsets[b_idx] = subs

    best_blocks = []
    best_count = float('inf')
    
    # Multiple runs
    runs = 0
    while time.time() - start_time < DEADLINE_SECONDS - 10:
        runs += 1
        # Randomized Greedy
        uncovered = full_mask
        current_blocks = []
        
        while uncovered:
            # Pick a random set bit in uncovered
            found_idx = -1
            attempts = 0
            while found_idx == -1 and attempts < 100:
                idx = random.randint(0, n_t - 1)
                if (uncovered >> idx) & 1:
                    found_idx = idx
                attempts += 1
            
            if found_idx == -1:
                # Fallback to linear scan if random fails
                for idx in range(n_t):
                    if (uncovered >> idx) & 1:
                        found_idx = idx
                        break
            
            if found_idx == -1:
                break # Should not happen if uncovered is non-zero
            
            # Get candidates
            candidates = blocks_containing[found_idx]
            
            best_gain = -1
            best_block_idx = -1
            # Randomize tie-breaking
            random.shuffle(candidates)
            
            for b_idx in candidates:
                gain = (block_masks[b_idx] & uncovered).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx != -1:
                current_blocks.append(best_block_idx)
                uncovered &= ~block_masks[best_block_idx]
        
        # Pruning
        current_blocks = _prune(current_blocks, block_subsets)
        
        # 2-Exchange Local Search
        current_blocks = _local_search_2exchange(current_blocks, block_subsets, block_masks, blocks_containing, n_t, full_mask, start_time)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]
            
        # If we hit a very good number, we can stop early?
        if best_count <= 334:
            break

    # Convert indices back to blocks
    result = [list(all_blocks[i]) for i in best_blocks]
    return result

def _prune(blocks, block_subsets):
    if not blocks:
        return []
    
    current = blocks[:]
    changed = True
    while changed:
        changed = False
        # Count coverage for each subset
        counts = {}
        for b_idx in current:
            for s_idx in block_subsets[b_idx]:
                counts[s_idx] = counts.get(s_idx, 0) + 1
        
        new_current = []
        for b_idx in current:
            # Check if any subset in this block is only covered by this block
            is_redundant = True
            for s_idx in block_subsets[b_idx]:
                if counts.get(s_idx, 0) == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                new_current.append(b_idx)
        
        if len(new_current) < len(current):
            current = new_current
            changed = True
    
    return current

def _local_search_2exchange(blocks, block_subsets, block_masks, blocks_containing, n_t, full_mask, start_time):
    current = blocks[:]
    improved = True
    while improved and (time.time() - start_time) < DEADLINE_SECONDS - 5:
        improved = False
        # Compute coverage counts
        counts = {}
        for b_idx in current:
            for s_idx in block_subsets[b_idx]:
                counts[s_idx] = counts.get(s_idx, 0) + 1
        
        # Try removing each block
        for i in range(len(current)):
            b_remove = current[i]
            # Check if removing b_remove leaves any subset uncovered
            can_remove = True
            for s_idx in block_subsets[b_remove]:
                if counts.get(s_idx, 0) == 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove it
                new_current = current[:i] + current[i+1:]
                # Update counts
                for s_idx in block_subsets[b_remove]:
                    counts[s_idx] -= 1
                current = new_current
                improved = True
                break
        
        if improved:
            continue
        
        # Try 2-exchange: remove one block, add another
        for i in range(len(current)):
            b_remove = current[i]
            # Find subsets that would become uncovered
            uncovered_subsets = []
            for s_idx in block_subsets[b_remove]:
                if counts.get(s_idx, 0) == 1:
                    uncovered_subsets.append(s_idx)
            
            if not uncovered_subsets:
                continue
            
            # Find a block that covers all uncovered_subsets
            best_add = -1
            best_gain = -1
            # Candidates: blocks that cover at least one uncovered subset
            candidate_set = set()
            for s_idx in uncovered_subsets:
                candidate_set.update(blocks_containing[s_idx])
            
            for b_add in candidate_set:
                if b_add in current:
                    continue
                # Check if b_add covers all uncovered_subsets
                covers_all = True
                for s_idx in uncovered_subsets:
                    if s_idx not in block_subsets[b_add]:
                        covers_all = False
                        break
                if covers_all:
                    # Calculate gain (number of new subsets covered)
                    gain = 0
                    for s_idx in block_subsets[b_add]:
                        if counts.get(s_idx, 0) == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_add = b_add
            
            if best_add != -1:
                # Perform the exchange
                new_current = current[:i] + current[i+1:] + [best_add]
                # Update counts
                for s_idx in block_subsets[b_remove]:
                    counts[s_idx] -= 1
                for s_idx in block_subsets[best_add]:
                    counts[s_idx] = counts.get(s_idx, 0) + 1
                current = new_current
                improved = True
                break
    
    return current
