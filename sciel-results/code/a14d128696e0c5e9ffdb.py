import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Map t-subsets to indices
    # We need to efficiently identify which t-subsets are in a block
    # Precompute list of t-subsets for each possible k-block? 
    # There are C(19,8) = 75582 blocks.
    # Each has C(8,5) = 56 t-subsets.
    # Total 4.2M pairs. Storing this might be memory heavy but fast.
    # Alternatively, compute on the fly.
    
    # Let's use a mapping from tuple(combination) to index
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_to_idx = {subset: i for i, subset in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    # Precompute for each possible block (k-subset) the list of t-subset indices it covers
    # This is expensive to generate all C(19,8) blocks initially?
    # 75582 blocks * 56 subsets = 4.2M operations. This is fast in Python if optimized?
    # Maybe too slow if done naively in a loop.
    # Let's try to generate them lazily or just use the greedy approach first.
    
    # Greedy Construction
    uncovered = set(range(num_t_subsets))
    blocks = []
    
    # For greedy, we need to find the block that covers the most uncovered t-subsets.
    # We can iterate through all possible blocks? 75k blocks.
    # For each block, count how many of its 56 t-subsets are in 'uncovered'.
    # This is 75k * 56 checks per step.
    # If we need ~300 blocks, that's 300 * 75k * 56 ~ 1.26 billion operations. Too slow in Python.
    
    # Better Greedy:
    # Pick a random uncovered t-subset.
    # Find a block containing it that covers the most OTHER uncovered t-subsets.
    # Candidates: blocks containing that specific t-subset.
    # How many blocks contain a specific t-subset?
    # We need to choose k-t = 8-5=3 more points from v-t = 19-5=14 points.
    # C(14,3) = 364 blocks per t-subset.
    # This is much smaller.
    
    # Precompute: for each t-subset index, list of block indices (or just the blocks themselves) that contain it?
    # Or just generate them on the fly.
    
    # Let's precompute the list of t-subset indices for each potential block is hard without iterating all blocks.
    # Instead, let's iterate all blocks once to build a reverse index?
    # block_to_t_indices = {}
    # for block in itertools.combinations(range(v), k):
    #     t_indices = [t_to_idx[t] for t in itertools.combinations(block, t)]
    #     block_to_t_indices[frozenset(block)] = t_indices
    # This takes ~4.2M inner loop iterations. Might take a few seconds.
    
    # Let's try to do it efficiently.
    if time.time() > deadline:
        return blocks # Empty if too slow, but shouldn't happen
    
    # Build reverse index: t_idx -> list of blocks (as frozensets) containing it?
    # Memory: 11628 entries, each with ~364 blocks. 11628 * 364 ~ 4.2M references.
    # This is feasible.
    
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    all_blocks = list(itertools.combinations(range(v), k))
    # Map block tuple to index for quick lookup?
    block_to_t_indices = {}
    
    for b in all_blocks:
        b_set = frozenset(b)
        t_indices = [t_to_idx[t] for t in itertools.combinations(b, t)]
        block_to_t_indices[b_set] = t_indices
        for idx in t_indices:
            t_to_blocks[idx].append(b_set)
            
    # Greedy Phase
    uncovered = set(range(num_t_subsets))
    current_blocks = set()
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-subset to break symmetry / speed up
        target_t = next(iter(uncovered))
        candidates = t_to_blocks[target_t]
        
        best_block = None
        best_gain = -1
        
        for b in candidates:
            # Count how many t-subsets in b are in uncovered
            gain = 0
            for idx in block_to_t_indices[b]:
                if idx in uncovered:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = b
            if time.time() > deadline: break
        if time.time() > deadline: break
        
        if best_block:
            current_blocks.add(best_block)
            # Update uncovered
            for idx in block_to_t_indices[best_block]:
                uncovered.discard(idx)
            
    # Convert to list for local search
    blocks_list = list(current_blocks)
    
    # Local Search Phase
    # State: set of blocks
    # Metric: len(blocks)
    # Constraint: all t-subsets covered
    
    # Precompute coverage counts
    cov_count = [0] * num_t_subsets
    for b in blocks_list:
        for idx in block_to_t_indices[b]:
            cov_count[idx] += 1
            
    # Check validity (should be valid from greedy)
    # Find safe blocks: blocks where all t-subsets have cov_count > 1
    # We want to remove safe blocks.
    
    last_improvement = time.time()
    
    while time.time() < deadline:
        # Find a safe block
        safe_block = None
        # Heuristic: prefer block with highest redundancy?
        # Just find any safe block
        for b in blocks_list:
            is_safe = True
            for idx in block_to_t_indices[b]:
                if cov_count[idx] == 1:
                    is_safe = False
                    break
            if is_safe:
                safe_block = b
                break
        
        if safe_block:
            # Remove it
            blocks_list.remove(safe_block)
            for idx in block_to_t_indices[safe_block]:
                cov_count[idx] -= 1
            last_improvement = time.time()
        else:
            # No safe block. Try a swap.
            # Pick a random block to remove
            if not blocks_list:
                break
                
            b_old = random.choice(blocks_list)
            # Identify subsets that become uncovered
            lost_indices = []
            for idx in block_to_t_indices[b_old]:
                if cov_count[idx] == 1:
                    lost_indices.append(idx)
            
            if not lost_indices:
                # This shouldn't happen if b_old was not safe, but just in case
                # If no lost indices, it was safe.
                blocks_list.remove(b_old)
                for idx in block_to_t_indices[b_old]:
                    cov_count[idx] -= 1
                last_improvement = time.time()
                continue
            
            # Find a replacement block b_new that covers all lost_indices
            # b_new must not be in blocks_list
            best_new = None
            # Candidates: blocks that cover the first lost index
            candidates = t_to_blocks[lost_indices[0]]
            
            for b_new in candidates:
                if b_new in blocks_list:
                    continue
                # Check if b_new covers all lost_indices
                new_t_set = set(block_to_t_indices[b_new])
                if all(idx in new_t_set for idx in lost_indices):
                    best_new = b_new
                    break
            if time.time() > deadline: break
            
            if best_new:
                # Perform swap
                blocks_list.remove(b_old)
                for idx in block_to_t_indices[b_old]:
                    cov_count[idx] -= 1
                
                blocks_list.append(best_new)
                for idx in block_to_t_indices[best_new]:
                    cov_count[idx] += 1
                
                # After swap, try to remove safe blocks immediately
                # (Loop back to top will do this, but we can do a quick pass)
                # Actually, the main loop will handle it.
                # To avoid infinite loops of swaps that don't reduce size, 
                # we might need to track history or accept that swaps keep size same 
                # but change structure.
                # If we swap and size doesn't change, we might cycle.
                # Add a small penalty or randomization?
                # For now, rely on finding safe blocks after swap.
            else:
                # No swap found. Try to add a block? 
                # If we are stuck, maybe we need to add a block to enable removal of 2?
                # This is getting complex. 
                # Let's try a random perturbation: Remove a random block, 
                # then greedily add blocks to restore validity? 
                # This increases size, so only do if we haven't improved in a while.
                if time.time() - last_improvement > 5:
                    # Perturbation
                    b_remove = random.choice(blocks_list)
                    blocks_list.remove(b_remove)
                    for idx in block_to_t_indices[b_remove]:
                        cov_count[idx] -= 1
                    
                    # Restore validity greedily
                    uncovered_now = set(i for i in range(num_t_subsets) if cov_count[i] == 0)
                    while uncovered_now:
                        target_t = next(iter(uncovered_now))
                        candidates = t_to_blocks[target_t]
                        best_block = None
                        best_gain = -1
                        for b in candidates:
                            if b in blocks_list: continue
                            gain = sum(1 for idx in block_to_t_indices[b] if idx in uncovered_now)
                            if gain > best_gain:
                                best_gain = gain
                                best_block = b
                        if best_block:
                            blocks_list.append(best_block)
                            for idx in block_to_t_indices[best_block]:
                                cov_count[idx] += 1
                                uncovered_now.discard(idx)
                        else:
                            # Should not happen
                            break
                        if time.time() > deadline: break
                    last_improvement = time.time()

    return [list(b) for b in blocks_list]
