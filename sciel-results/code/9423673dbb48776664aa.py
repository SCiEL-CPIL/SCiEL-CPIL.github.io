import itertools
import random
import time
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    n_t = len(all_t_subsets)
    
    # Map each t-subset to an index for faster counting
    t_index = {t_set: i for i, t_set in enumerate(all_t_subsets)}
    
    # Precompute for each block (as a frozenset) the list of t-subset indices it covers
    # But generating all C(19,8) blocks upfront is 75582, which is fine
    # However, we'll generate blocks on the fly and cache their coverage
    
    # For greedy, we need to efficiently find the best block
    # Strategy: maintain uncovered set, and for each candidate block compute gain
    
    # Precompute all possible k-blocks and their t-coverage
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Precompute t-coverage for each block
    block_t_indices = []
    for block in all_blocks:
        t_subs = list(itertools.combinations(block, t))
        indices = [t_index[ts] for ts in t_subs]
        block_t_indices.append(indices)
    
    # Greedy phase with randomization
    uncovered = set(range(n_t))  # Use indices for speed
    blocks_selected = []
    counts = [0] * n_t
    
    # For efficiency, we'll use a different approach:
    # Instead of iterating all blocks each time, we can be smarter
    
    # Randomized greedy with multiple restarts
    best_solution = None
    best_count = float('inf')
    
    # Number of restarts based on time budget
    # Each greedy run takes roughly O(n_t * n_blocks * t_in_block) which is expensive
    # Let's do a single careful greedy with good heuristics, then local search
    
    # Better approach: weighted greedy
    # Assign weights to uncovered t-subsets, pick block that maximizes weighted gain
    
    # Let's do a greedy that picks blocks in a smart order
    # Use a priority queue or just iterate
    
    # For C(19,8,5): n_t = 11628, n_blocks = 75582
    # Each block covers C(8,5) = 56 t-subsets
    
    # Greedy: repeatedly pick the block with maximum uncovered coverage
    # This is O(n_blocks * 56) per iteration, and we do ~300 iterations
    # Total: 300 * 75582 * 56 ≈ 1.27 billion operations - too slow in Python
    
    # Need a smarter approach. Let's use the fact that we can:
    # 1. Start with a random set of blocks
    # 2. Use local search to improve
    
    # Alternative: use the structure of the problem
    # A block is an 8-subset. We can represent it as a bitmask of 19 bits.
    
    # Let's try a different greedy: 
    # Pick a random uncovered t-subset, then find the best block containing it
    # This is what the original does, but we can make it faster
    
    # Precompute for each t-subset, which blocks contain it
    t_to_blocks = [[] for _ in range(n_t)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_to_blocks[idx].append(b_idx)
    
    # Now greedy: pick random uncovered t-subset, find best block containing it
    random.seed(42)
    
    uncovered_set = set(range(n_t))
    selected_blocks = []
    counts = [0] * n_t
    
    while uncovered_set:
        if time.time() > deadline - 30:  # Reserve 30s for local search
            break
        
        # Pick a random uncovered t-subset
        target = random.choice(list(uncovered_set))
        
        # Find the best block containing this t-subset
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in t_to_blocks[target]:
            # Compute gain: number of uncovered t-subsets covered by this block
            gain = 0
            for idx in block_t_indices[b_idx]:
                if idx in uncovered_set:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
        
        selected_blocks.append(best_block_idx)
        
        # Update uncovered and counts
        for idx in block_t_indices[best_block_idx]:
            counts[idx] += 1
            if counts[idx] == 1:
                uncovered_set.discard(idx)
    
    # Convert to actual blocks
    blocks = [list(all_blocks[i]) for i in selected_blocks]
    
    # Local search: try to remove redundant blocks
    max_removal_rounds = 10
    for _ in range(max_removal_rounds):
        if time.time() > deadline - 5:
            break
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            block = blocks[i]
            # Check if block is redundant
            redundant = True
            for t_sub in itertools.combinations(block, t):
                idx = t_index[t_sub]
                if counts[idx] <= 1:
                    redundant = False
                    break
            
            if redundant:
                blocks.pop(i)
                for t_sub in itertools.combinations(block, t):
                    idx = t_index[t_sub]
                    counts[idx] -= 1
                changed = True
        if not changed:
            break
    
    # Additional local search: try swapping blocks
    # For each block, try replacing it with a different block that covers the same critical t-subsets
    if time.time() < deadline - 10:
        for _ in range(5):  # 5 rounds of swap attempts
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(len(blocks)):
                if time.time() > deadline - 5:
                    break
                block = blocks[i]
                # Find t-subsets that are only covered by this block
                critical = []
                for t_sub in itertools.combinations(block, t):
                    idx = t_index[t_sub]
                    if counts[idx] == 1:
                        critical.append(idx)
                
                if not critical:
                    continue
                
                # Try to find a replacement block that covers all critical t-subsets
                # and doesn't add too many new blocks
                # This is expensive, so let's just try a few random blocks
                for _ in range(20):
                    if time.time() > deadline - 5:
                        break
                    new_block_idx = random.randint(0, n_blocks - 1)
                    new_block = all_blocks[new_block_idx]
                    
                    # Check if new block covers all critical t-subsets
                    new_block_t_set = set(block_t_indices[new_block_idx])
                    if all(c in new_block_t_set for c in critical):
                        # Check if this swap is beneficial
                        # Count how many t-subsets would become uncovered
                        # and how many would be newly covered
                        would_uncover = 0
                        for t_sub in itertools.combinations(block, t):
                            idx = t_index[t_sub]
                            if counts[idx] == 1 and idx not in new_block_t_set:
                                would_uncover += 1
                        
                        if would_uncover == 0:
                            # Swap is safe
                            # Check if it reduces total count
                            blocks[i] = new_block
                            # Update counts
                            for t_sub in itertools.combinations(block, t):
                                idx = t_index[t_sub]
                                counts[idx] -= 1
                            for t_sub in itertools.combinations(new_block, t):
                                idx = t_index[t_sub]
                                counts[idx] += 1
                            improved = True
                            break
            if not improved:
                break
    
    # Final removal pass
    for _ in range(5):
        if time.time() > deadline - 2:
            break
        changed = False
        for i in range(len(blocks) - 1, -1, -1):
            if time.time() > deadline - 2:
                break
            block = blocks[i]
            redundant = True
            for t_sub in itertools.combinations(block, t):
                idx = t_index[t_sub]
                if counts[idx] <= 1:
                    redundant = False
                    break
            if redundant:
                blocks.pop(i)
                for t_sub in itertools.combinations(block, t):
                    idx = t_index[t_sub]
                    counts[idx] -= 1
                changed = True
        if not changed:
            break
    
    return blocks
