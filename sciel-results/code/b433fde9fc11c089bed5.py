"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def _precompute(v, k, t):
    """
    Pre-compute all t-subsets, map them to indices, and create bitmasks for all k-blocks.
    Returns:
    - all_t_subsets: list of t-subsets
    - t_index_map: dict from t-subset to index
    - all_blocks: list of k-blocks
    - block_bitmasks: list of integers representing coverage of each block
    - num_subsets: total number of t-subsets
    - full_mask: integer with all bits set for num_subsets
    """
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_index_map = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    full_mask = (1 << num_subsets) - 1
    
    all_blocks = list(itertools.combinations(range(v), k))
    block_bitmasks = []
    
    for block in all_blocks:
        mask = 0
        # Since t is small (4) and k is small (7), we can iterate combinations
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            mask |= (1 << idx)
        block_bitmasks.append(mask)
        
    return all_t_subsets, t_index_map, all_blocks, block_bitmasks, num_subsets, full_mask

def _greedy_cover_from_state(uncovered_mask, available_blocks, block_bitmasks, all_blocks, max_iter=1000):
    """
    Greedily select blocks to cover the remaining uncovered_mask.
    available_blocks: list of indices into all_blocks
    Returns list of block tuples.
    """
    selected_blocks = []
    current_uncovered = uncovered_mask
    
    for _ in range(max_iter):
        if current_uncovered == 0:
            break
            
        best_idx = -1
        best_gain = 0
        
        # Find block with max gain
        for idx in available_blocks:
            gain = bin(current_uncovered & block_bitmasks[idx]).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = idx
                
        if best_idx == -1 or best_gain == 0:
            # Should not happen if full_mask is reachable
            break
            
        selected_blocks.append(all_blocks[best_idx])
        current_uncovered &= ~block_bitmasks[best_idx]
        
    return selected_blocks

def _reduce_covering(blocks, block_bitmasks_map, full_mask):
    """
    Remove redundant blocks from a list of blocks.
    block_bitmasks_map: dict mapping block tuple to its bitmask.
    """
    if not blocks:
        return blocks
    
    # Calculate total coverage
    total_mask = 0
    for b in blocks:
        total_mask |= block_bitmasks_map[b]
        
    # If total coverage is not full, we can't reduce safely without checking validity, 
    # but the reduction step assumes we start with a valid cover.
    
    n = len(blocks)
    keep = [True] * n
    
    # Optimization: If a block's mask is a subset of another block's mask, it's redundant
    # But stronger: If removing block i doesn't change total_mask, it's redundant.
    
    # To speed up, we can iterate multiple times or just once. Once is O(N).
    # Let's do a single pass removal.
    new_blocks = []
    current_total = total_mask
    
    # Sort blocks by "uniqueness" or just iterate. 
    # A block is redundant if (total_mask & ~mask_i) == total_mask
    # i.e., total_mask | mask_i == total_mask (which is always true)
    # Wait, redundancy check: 
    # Let U be the union of all blocks.
    # Block i is redundant if U \ {i} still covers U.
    # This is equivalent to: for every subset s in mask_i, s is covered by some other block j != i.
    # Or simply: (total_mask_without_i) == total_mask.
    
    # Efficient way:
    # Pre-compute masks for each block in the list
    masks = [block_bitmasks_map[b] for b in blocks]
    
    # We can try to remove blocks.
    # A simple heuristic: iterate through blocks, if removing it doesn't change the union of the *rest*, remove it.
    # But calculating union of rest is O(N) per check -> O(N^2).
    # With N~150, N^2=22500. Bit ops are fast.
    
    # Let's use the property: 
    # If mask_i is a subset of (union of all masks except i), then i is redundant.
    
    # Alternative: 
    # Calculate total_mask.
    # For each i, check if (total_mask & ~masks[i]) is still covered by others? 
    # No, that's not right.
    
    # Correct check:
    # Let U = OR(all masks).
    # Block i is redundant if OR(all masks except i) == U.
    
    # We can compute prefix and suffix ORs to make this O(N).
    # prefix_or[i] = OR(masks[0..i-1])
    # suffix_or[i] = OR(masks[i+1..N-1])
    # Then union_except_i = prefix_or[i] | suffix_or[i]
    # If union_except_i == total_mask, then i is redundant.
    
    prefix_or = [0] * (n + 1)
    for i in range(n):
        prefix_or[i+1] = prefix_or[i] | masks[i]
        
    suffix_or = [0] * (n + 1)
    for i in range(n-1, -1, -1):
        suffix_or[i] = suffix_or[i+1] | masks[i]
        
    for i in range(n):
        union_except_i = prefix_or[i] | suffix_or[i+1]
        if union_except_i == total_mask:
            keep[i] = False
        else:
            keep[i] = True
            
    return [b for b, k in zip(blocks, keep) if k]

def _crossover(parent1, parent2, all_blocks, block_bitmasks, full_mask, t_index_map, v, k, t):
    """
    Merge two coverings by selecting blocks that cover the most uncovered 4-subsets.
    """
    # Union of blocks from parents
    pool_set = set(parent1 + parent2)
    pool = list(pool_set)
    
    if not pool:
        return []
        
    # Get bitmasks for blocks in the pool
    # We need a map from block to bitmask. 
    # Since all_blocks is the universe, we can create a map for the pool.
    # However, block_bitmasks is a list aligned with all_blocks.
    # We need to find the index in all_blocks for each block in the pool.
    # This is slow if done naively. 
    # Let's assume the caller provides a map or we use a global map.
    # For efficiency in this function, let's pass a map.
    # Actually, _precompute returns all_blocks. We can create a global dict outside.
    
    # To keep the interface simple, I will construct the map here if not provided, 
    # but it's better to pass it. 
    # Let's modify the signature to accept block_to_mask_map.
    
    # For now, let's assume we can look up masks.
    # I will update the main construct function to pass the map.
    
    pass # Placeholder

def construct(v, k, t):
    """Main entry point."""
    start_time = time.time()
    time_limit = 115.0
    
    # Pre-compute
    all_t_subsets, t_index_map, all_blocks, block_bitmasks, num_subsets, full_mask = _precompute(v, k, t)
    
    # Create a fast lookup map for block tuple -> bitmask
    block_to_mask = {block: mask for block, mask in zip(all_blocks, block_bitmasks)}
    
    # Helper to get mask from block
    def get_mask(block):
        return block_to_mask.get(block, 0)
    
    # 1. Initial Population
    population = []
    
    # Greedy 1
    uncovered = full_mask
    blocks1 = []
    while uncovered:
        best_idx = -1
        best_gain = 0
        # Randomize start index to vary the greedy path?
        # For diversity, we can shuffle the candidate order or just pick best.
        # Picking best is deterministic. Let's add some noise.
        # We'll just run standard greedy.
        for i, mask in enumerate(block_bitmasks):
            gain = bin(uncovered & mask).count('1')
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        if best_idx == -1:
            break
        blocks1.append(all_blocks[best_idx])
        uncovered &= ~block_bitmasks[best_idx]
        
    population.append(blocks1)
    
    # Random + Repair (Greedy completion)
    for _ in range(4):
        # Start with some random blocks
        rand_blocks = [random.choice(all_blocks) for _ in range(30)]
        # Compute current coverage
        current_mask = 0
        for b in rand_blocks:
            current_mask |= get_mask(b)
        uncovered = full_mask & ~current_mask
        
        # Greedily add blocks to cover uncovered
        added = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i, mask in enumerate(block_bitmasks):
                gain = bin(uncovered & mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            added.append(all_blocks[best_idx])
            uncovered &= ~block_bitmasks[best_idx]
            
        candidate = rand_blocks + added
        # Reduce
        candidate = _reduce_covering(candidate, block_to_mask, full_mask)
        population.append(candidate)
        
    # Ensure population is not empty
    if not population:
        # Fallback: pure greedy
        return blocks1
        
    best_individual = min(population, key=len)
    
    # 2. Genetic Loop
    while time.time() - start_time < time_limit:
        population.sort(key=len)
        
        # Elitism
        elite = population[:max(2, len(population)//2)]
        
        new_population = list(elite)
        
        # Generate children
        target_size = len(population)
        
        while len(new_population) < target_size:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            # Crossover: Merge and Greedy Select
            pool = list(set(p1 + p2))
            
            # Sort pool by "quality" (coverage size) to help greedy? 
            # Or just greedy select based on uncovered.
            
            uncovered = full_mask
            child_blocks = []
            # To make it faster, we can limit the number of blocks we consider in the first pass
            # or just iterate through the pool.
            
            # Greedy selection from pool
            # We need to find the block in the pool that covers the most uncovered.
            # Pool size is small (< 300).
            
            # Pre-compute masks for pool
            pool_masks = [(b, get_mask(b)) for b in pool]
            
            while uncovered:
                best_block = None
                best_gain = 0
                for b, m in pool_masks:
                    gain = bin(uncovered & m).count('1')
                    if gain > best_gain:
                        best_gain = gain
                        best_block = b
                        best_mask = m
                
                if best_block is None or best_gain == 0:
                    # Pool is insufficient, need to add from global
                    # Find best global block
                    global_best_idx = -1
                    global_best_gain = 0
                    for i, mask in enumerate(block_bitmasks):
                        gain = bin(uncovered & mask).count('1')
                        if gain > global_best_gain:
                            global_best_gain = gain
                            global_best_idx = i
                    if global_best_idx == -1:
                        break
                    best_block = all_blocks[global_best_idx]
                    best_mask = block_bitmasks[global_best_idx]
                    
                child_blocks.append(best_block)
                uncovered &= ~best_mask
                
            # Reduce child
            child_blocks = _reduce_covering(child_blocks, block_to_mask, full_mask)
            
            # Mutation: Randomly replace a block
            if random.random() < 0.3 and child_blocks:
                idx = random.randint(0, len(child_blocks)-1)
                child_blocks[idx] = random.choice(all_blocks)
                child_blocks = _reduce_covering(child_blocks, block_to_mask, full_mask)
                
            new_population.append(child_blocks)
            
        population = new_population
        
        current_best = min(population, key=len)
        if len(current_best) < len(best_individual):
            best_individual = current_best
            if len(best_individual) <= 151:
                break
                
    # Final Reduction
    best_individual = _reduce_covering(best_individual, block_to_mask, full_mask)
    
    # Validation
    # Check if valid
    total_mask = 0
    for b in best_individual:
        total_mask |= get_mask(b)
        
    if total_mask != full_mask:
        # Fallback to greedy if invalid (should not happen)
        uncovered = full_mask
        fallback = []
        while uncovered:
            best_idx = -1
            best_gain = 0
            for i, mask in enumerate(block_bitmasks):
                gain = bin(uncovered & mask).count('1')
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i
            if best_idx == -1:
                break
            fallback.append(all_blocks[best_idx])
            uncovered &= ~block_bitmasks[best_idx]
        best_individual = fallback
        
    return best_individual
# EVOLVE-BLOCK-END
