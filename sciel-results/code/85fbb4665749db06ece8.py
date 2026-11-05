"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def _get_all_t_subsets(v, t):
    """Generate all t-subsets and map them to indices."""
    subsets = list(itertools.combinations(range(v), t))
    mapping = {s: i for i, s in enumerate(subsets)}
    return subsets, mapping

def _compute_block_mask(block, t_index_map, num_subsets):
    """Compute boolean mask for a single block."""
    mask = np.zeros(num_subsets, dtype=np.uint8)
    for combo in itertools.combinations(sorted(block), 4):
        idx = t_index_map[combo]
        mask[idx] = 1
    return mask

def _greedy_cover(v, k, t, time_limit=10.0):
    """Fast greedy covering using precomputed block masks."""
    start_time = time.time()
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Pre-compute coverage matrix as uint8 for faster matmul
    coverage_matrix = np.zeros((n_blocks, num_subsets), dtype=np.uint8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
    
    uncovered = np.ones(num_subsets, dtype=np.uint8)
    blocks = []
    
    while uncovered.any():
        # Vectorized gain computation
        gains = coverage_matrix @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        
        blocks.append(list(all_blocks[best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        
        if time.time() - start_time > time_limit:
            break
    
    return blocks

def _repair(v, k, t, blocks, time_limit=5.0):
    """Repair and reduce a list of blocks using count-based redundancy detection."""
    start_time = time.time()
    
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    
    # Compute per-subset coverage counts
    cover_count = np.zeros(num_subsets, dtype=np.uint16)
    block_masks = []
    
    for b in blocks:
        mask = _compute_block_mask(b, t_index_map, num_subsets)
        block_masks.append(mask)
        cover_count += mask
    
    uncovered = cover_count == 0
    
    # Greedy add to cover remaining
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        first_idx = int(np.argmax(uncovered))
        first_subset = all_subsets[first_idx]
        
        # Find best block containing first_subset
        remaining_points = [x for x in range(v) if x not in first_subset]
        best_block = None
        best_gain = -1
        
        for candidate_extra in itertools.combinations(remaining_points, k - t):
            block = tuple(sorted(list(first_subset) + list(candidate_extra)))
            gain = 0
            for combo in itertools.combinations(block, t):
                idx = t_index_map[combo]
                if cover_count[idx] == 0:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block = block
        
        if best_block is None:
            break
        
        blocks.append(list(best_block))
        new_mask = _compute_block_mask(best_block, t_index_map, num_subsets)
        block_masks.append(new_mask)
        cover_count += new_mask
        uncovered = cover_count == 0
    
    # Reduction: remove redundant blocks
    # A block is redundant if for every 4-subset it covers, another block also covers it
    # i.e., no 4-subset has cover_count == 1 that is covered by this block
    n_blocks = len(blocks)
    if n_blocks <= 1:
        return blocks
    
    # Recompute cover_count from scratch for accuracy after additions
    cover_count = np.zeros(num_subsets, dtype=np.uint16)
    for mask in block_masks:
        cover_count += mask
    
    keep = [True] * n_blocks
    changed = True
    while changed and (time.time() - start_time) < time_limit:
        changed = False
        for i in range(n_blocks):
            if not keep[i]:
                continue
            # Check if block i is redundant:
            # For all subsets covered by block i, cover_count > 1 (i.e., some other block covers it)
            # Equivalently: (block_masks[i] == 1) & (cover_count == 1) should be empty
            exclusive = block_masks[i] & (cover_count == 1)
            if exclusive.sum() == 0:
                keep[i] = False
                cover_count -= block_masks[i]
                changed = True
    
    return [b for b, k_flag in zip(blocks, keep) if k_flag]

def _crossover(parent1, parent2, v, k, t):
    """Merge two coverings via greedy selection from union pool."""
    pool = list(set([tuple(b) for b in parent1 + parent2]))
    
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    
    pool_masks = []
    for b in pool:
        mask = _compute_block_mask(b, t_index_map, num_subsets)
        pool_masks.append(mask)
    
    uncovered = np.ones(num_subsets, dtype=np.uint8)
    child_blocks = []
    available = set(range(len(pool)))
    
    while uncovered.any() and available:
        best_idx = -1
        best_gain = -1
        
        for i in available:
            gain = int(np.sum(pool_masks[i] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        child_blocks.append(list(pool[best_idx]))
        uncovered[pool_masks[best_idx] == 1] = 0
        available.discard(best_idx)
    
    if uncovered.any():
        child_blocks = _repair(v, k, t, child_blocks)
    
    return child_blocks

def construct(v, k, t):
    """Main entry point."""
    start_time = time.time()
    time_limit = 115.0
    
    # Initial population with diversity
    greedy1 = _greedy_cover(v, k, t, time_limit=8.0)
    
    population = [greedy1]
    
    # Add diverse random individuals
    for _ in range(6):
        n_rand = random.randint(40, 80)
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(n_rand)]
        pop_ind = _repair(v, k, t, rand_blocks, time_limit=3.0)
        population.append(pop_ind)
    
    best_individual = min(population, key=len)
    best_len = len(best_individual)
    
    while time.time() - start_time < time_limit:
        population.sort(key=len)
        
        # Elitism: keep top half
        elite_size = max(2, len(population) // 2)
        elite = population[:elite_size]
        
        new_population = list(elite)
        
        # Generate children to fill population
        target_size = len(population)
        while len(new_population) < target_size:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t)
            
            # Mutation with probability 0.6
            if random.random() < 0.6 and child:
                n_mut = random.randint(1, 3)
                for _ in range(n_mut):
                    idx = random.randint(0, len(child) - 1)
                    child[idx] = sorted(random.sample(range(v), k))
                child = _repair(v, k, t, child, time_limit=2.0)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=len)
        if len(current_best) < best_len:
            best_individual = current_best
            best_len = len(best_individual)
            if best_len <= 151:
                break
    
    # Final repair
    final_blocks = _repair(v, k, t, best_individual, time_limit=5.0)
    
    # Validate
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    cover_count = np.zeros(len(all_subsets), dtype=np.uint16)
    for b in final_blocks:
        mask = _compute_block_mask(b, t_index_map, len(all_subsets))
        cover_count += mask
    
    if not np.all(cover_count > 0):
        # Fallback to full greedy
        final_blocks = _greedy_cover(v, k, t, time_limit=110.0)
    
    return final_blocks
# EVOLVE-BLOCK-END
