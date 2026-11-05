import itertools
import time
import random
import math
import sys

def construct(v, k, t):
    """
    Find a covering of t-subsets of v points using k-subsets (blocks).
    Returns a list of blocks (each is a sorted list of k distinct ints in 0..v-1).
    Every t-subset must be contained in at least one block.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    def check_time(budget=5.0):
        return time.time() - start_time > deadline - budget
    
    # Precompute all t-subsets and map to indices
    all_tsubs = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_tsubs)
    tsub_index = {ts: i for i, ts in enumerate(all_tsubs)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # For each block, precompute which t-subsets it covers
    # block_tsubs[b] = list of t-subset indices covered by block b
    block_tsubs = []
    for b in all_blocks:
        tsubs_in_b = []
        for ts in itertools.combinations(b, t):
            tsubs_in_b.append(tsub_index[ts])
        block_tsubs.append(tsubs_in_b)
    
    # For each t-subset, precompute which blocks cover it
    # tsub_blocks[ts] = list of block indices that cover t-subset ts
    tsub_blocks = [[] for _ in range(num_tsubs)]
    for b in range(num_blocks):
        for ts_idx in block_tsubs[b]:
            tsub_blocks[ts_idx].append(b)
    
    # Degrees
    degs = [len(tsub_blocks[i]) for i in range(num_tsubs)]
    for i in range(num_tsubs):
        if degs[i] == 0:
            degs[i] = 1
    
    # Base weights: 1/degree
    base_weights = [1.0 / degs[i] for i in range(num_tsubs)]
    
    def greedy_set_cover(weights):
        """Greedy set cover using bitmask for coverage tracking."""
        # Use a set of uncovered t-subsets
        uncovered = set(range(num_tsubs))
        selected = []
        
        # Precompute for each block: the set of t-subsets it covers (as a frozenset for fast intersection)
        # Actually, let's just iterate through tsubs in each block
        
        while uncovered:
            if check_time(2.0):
                break
            
            best_block = -1
            best_gain = 0
            
            for b in range(num_blocks):
                # Check if this block could improve
                # Count how many uncovered t-subsets it covers
                gain = 0
                for ts_idx in block_tsubs[b]:
                    if ts_idx in uncovered:
                        gain += 1
                        if gain >= best_gain:
                            # Early stop if we can't beat current best
                            # Actually we need exact count for tie-breaking, so no early stop
                            pass
                
                weighted_gain = sum(weights[ts_idx] for ts_idx in block_tsubs[b] if ts_idx in uncovered)
                
                if weighted_gain > best_gain:
                    best_gain = weighted_gain
                    best_block = b
            
            if best_block == -1 or best_gain <= 0:
                # Stuck
                break
            
            selected.append(best_block)
            
            # Remove covered t-subsets
            for ts_idx in block_tsubs[best_block]:
                uncovered.discard(ts_idx)
        
        return selected
    
    def reduce_blocks(block_indices):
        """Remove redundant blocks."""
        if not block_indices:
            return []
        
        # Count coverage
        cover_count = [0] * num_tsubs
        for b in block_indices:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
        
        changed = True
        while changed:
            if check_time(3.0):
                break
            changed = False
            new_blocks = []
            for b in block_indices:
                # Check if any t-subset in this block has coverage exactly 1
                essential = False
                for ts_idx in block_tsubs[b]:
                    if cover_count[ts_idx] == 1:
                        essential = True
                        break
                if essential:
                    new_blocks.append(b)
                else:
                    # Remove this block
                    changed = True
                    for ts_idx in block_tsubs[b]:
                        cover_count[ts_idx] -= 1
            block_indices = new_blocks
        
        return block_indices
    
    def local_search_remove(block_indices):
        """Try to remove blocks one at a time, in random order."""
        if not block_indices:
            return []
        
        block_indices = list(block_indices)
        random.shuffle(block_indices)
        
        cover_count = [0] * num_tsubs
        for b in block_indices:
            for ts_idx in block_tsubs[b]:
                cover_count[ts_idx] += 1
        
        new_blocks = []
        for b in block_indices:
            if check_time(2.0):
                new_blocks.append(b)
                continue
            
            # Check if removing this block keeps all covered
            can_remove = True
            for ts_idx in block_tsubs[b]:
                if cover_count[ts_idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                for ts_idx in block_tsubs[b]:
                    cover_count[ts_idx] -= 1
            else:
                new_blocks.append(b)
        
        return new_blocks
    
    # --- Strategy: Multiple greedy runs with different weight schemes, then reduce + local search ---
    
    best_blocks = []
    best_len = float('inf')
    
    # Run 1: Standard greedy with 1/deg weights
    if not check_time(10.0):
        blocks1 = greedy_set_cover(base_weights)
        blocks1 = reduce_blocks(blocks1)
        blocks1 = local_search_remove(blocks1)
        if len(blocks1) < best_len:
            best_len = len(blocks1)
            best_blocks = blocks1
    
    # Run 2-8: Greedy with perturbed weights
    num_runs = 7
    for run in range(num_runs):
        if check_time(10.0):
            break
        
        # Create perturbed weights
        weights_run = list(base_weights)
        for i in range(num_tsubs):
            # Add random noise
            noise = random.gauss(0, 0.15 * base_weights[i])
            weights_run[i] = max(0.01 * base_weights[i], base_weights[i] + noise)
        
        blocks_r = greedy_set_cover(weights_run)
        blocks_r = reduce_blocks(blocks_r)
        blocks_r = local_search_remove(blocks_r)
        
        if len(blocks_r) < best_len:
            best_len = len(blocks_r)
            best_blocks = blocks_r
    
    # Try a few more local search iterations on the best
    for iteration in range(3):
        if check_time(5.0):
            break
        improved = local_search_remove(best_blocks)
        if len(improved) < best_len:
            best_len = len(improved)
            best_blocks = improved
        else:
            break
    
    # Final reduction pass
    best_blocks = reduce_blocks(best_blocks)
    best_len = len(best_blocks)
    
    return [list(all_blocks[b]) for b in best_blocks]
