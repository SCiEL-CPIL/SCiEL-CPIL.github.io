import itertools
import random
import time
import sys

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Greedy + Local Search.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Precomputation
    # 1. Generate all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_set_to_idx = {ts: i for i, ts in enumerate(t_subsets)}
    num_tsets = len(t_subsets)
    
    # 2. Generate all k-subsets (blocks) and map to indices
    k_subsets = list(itertools.combinations(range(v), k))
    num_ksets = len(k_subsets)
    
    # 3. Map each block to the list of t-set indices it covers
    block_to_tsets = [[] for _ in range(num_ksets)]
    # 4. Map each t-set to the list of block indices that cover it
    tset_to_blocks = [[] for _ in range(num_tsets)]
    
    for b_idx, block in enumerate(k_subsets):
        block_set = set(block)
        # Find all t-subsets of this block
        for ts in itertools.combinations(block, t):
            ts_idx = t_set_to_idx[ts]
            block_to_tsets[b_idx].append(ts_idx)
            tset_to_blocks[ts_idx].append(b_idx)
            
    # State
    current_blocks = []  # List of block indices
    tset_counts = [0] * num_tsets
    in_current = [False] * num_ksets
    
    def add_block(b_idx):
        current_blocks.append(b_idx)
        in_current[b_idx] = True
        for ts_idx in block_to_tsets[b_idx]:
            tset_counts[ts_idx] += 1
            
    def remove_block(b_idx):
        # Remove from current_blocks (O(n) but n is small ~150)
        current_blocks.remove(b_idx)
        in_current[b_idx] = False
        for ts_idx in block_to_tsets[b_idx]:
            tset_counts[ts_idx] -= 1

    # Greedy Initialization
    uncovered = set(range(num_tsets))
    
    while uncovered:
        best_b = -1
        max_gain = 0
        # Optimize: If uncovered is large, we can pick a random block from uncovered's covering blocks?
        # No, standard greedy is safer for initialization.
        # To speed up, we can iterate through blocks and check gain.
        # With 50k blocks, this is 50k * 35 ops per iteration. ~1.7M ops.
        # If we need ~150 blocks, that's 250M ops. Might be slow in Python.
        
        # Optimization: Pick a random uncovered t-set, get all blocks covering it, 
        # and find the one with max gain among those.
        # This is much faster.
        if uncovered:
            # Pick a random uncovered t-set to anchor the search
            # Using a deterministic min might be better for reproducibility, but random helps escape bad local minima in init?
            # Let's stick to finding the block that covers the most uncovered sets.
            # We can approximate by picking a random uncovered set and checking its blocks.
            anchor_ts = random.choice(list(uncovered))
            candidates = tset_to_blocks[anchor_ts]
            
            for b_idx in candidates:
                if in_current[b_idx]:
                    continue
                gain = 0
                for ts_idx in block_to_tsets[b_idx]:
                    if ts_idx in uncovered:
                        gain += 1
                if gain > max_gain:
                    max_gain = gain
                    best_b = b_idx
            
            if best_b == -1:
                # Should not happen if covering is possible
                break
                
            add_block(best_b)
            # Update uncovered
            for ts_idx in block_to_tsets[best_b]:
                if ts_idx in uncovered:
                    uncovered.remove(ts_idx)
                    
            if time.time() > deadline:
                break

    # Local Search
    # We want to minimize len(current_blocks)
    
    while time.time() < deadline:
        # Phase 1: Try to remove redundant blocks
        removed = False
        # Shuffle to avoid bias
        random.shuffle(current_blocks)
        
        for b_idx in list(current_blocks):
            if time.time() > deadline:
                break
            
            # Check if any t-set in this block has count 1
            is_redundant = True
            for ts_idx in block_to_tsets[b_idx]:
                if tset_counts[ts_idx] == 1:
                    is_redundant = False
                    break
            
            if is_redundant:
                remove_block(b_idx)
                removed = True
                break # Restart loop as state changed
                
        if removed:
            continue
            
        # Phase 2: Try to swap a critical block
        # Find all critical blocks
        critical_blocks = []
        for b_idx in current_blocks:
            is_critical = False
            for ts_idx in block_to_tsets[b_idx]:
                if tset_counts[ts_idx] == 1:
                    is_critical = True
                    break
            if is_critical:
                critical_blocks.append(b_idx)
                
        if not critical_blocks:
            # No critical blocks, but no redundant blocks? 
            # This implies all blocks are critical but none are redundant?
            # If a block is critical, it has a unique t-set. 
            # If it's not redundant, it has a unique t-set.
            # So we must swap.
            pass
            
        if critical_blocks:
            # Pick a random critical block
            old_b = random.choice(critical_blocks)
            
            # Find unique t-sets of old_b
            unique_ts = []
            for ts_idx in block_to_tsets[old_b]:
                if tset_counts[ts_idx] == 1:
                    unique_ts.append(ts_idx)
            
            # Find candidate new blocks that cover all unique_ts
            # Intersection of tset_to_blocks[ts] for ts in unique_ts
            if not unique_ts:
                continue
                
            # Start with the list for the first unique t-set
            candidates = set(tset_to_blocks[unique_ts[0]])
            for ts_idx in unique_ts[1:]:
                candidates.intersection_update(tset_to_blocks[ts_idx])
                if not candidates:
                    break
                    
            # Filter out blocks already in current
            candidates.difference_update(current_blocks)
            
            if candidates:
                # Pick a random candidate
                new_b = random.choice(list(candidates))
                
                # Perform swap
                remove_block(old_b)
                add_block(new_b)
                # Do not break, continue to check for removals in next iteration
                # But we should probably limit swaps to avoid cycling?
                # For now, let it run.
            else:
                # No swap found for this block. Try another?
                # To save time, just break and restart or try next critical block?
                # Let's just continue the outer loop, which will pick another critical block next time.
                pass
        else:
            # No critical blocks found? This means all t-sets have count >= 2?
            # Then any block is redundant. Phase 1 should have caught it.
            # Break to avoid infinite loop if logic is flawed.
            break

    # Convert block indices back to lists of points
    result = [list(k_subsets[b_idx]) for b_idx in current_blocks]
    return result

if __name__ == "__main__":
    # Test with the specific task parameters
    v, k, t = 19, 7, 4
    blocks = construct(v, k, t)
    print(f"Number of blocks: {len(blocks)}")
