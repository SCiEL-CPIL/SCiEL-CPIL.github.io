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

def _block_to_mask_indices(block, t_index_map, t):
    """Convert a block to a list of indices of covered t-subsets."""
    indices = []
    for combo in itertools.combinations(sorted(block), t):
        idx = t_index_map.get(combo)
        if idx is not None:
            indices.append(idx)
    return indices

def _greedy_cover(v, k, t, time_limit=10.0):
    """Fast greedy covering. Returns list of blocks."""
    start_time = time.time()
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    uncovered = np.ones(num_subsets, dtype=bool)
    blocks = []
    
    # Pre-compute all possible blocks
    all_blocks = list(itertools.combinations(range(v), k))
    
    # Pre-compute coverage masks for all blocks as sparse index lists
    # This is faster to compute and use than dense boolean matrices for this size
    block_indices = []
    for block in all_blocks:
        indices = []
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            indices.append(idx)
        block_indices.append(indices)
    
    # Greedy Set Cover using vectorized gain calculation
    # We can't easily vectorize the gain calculation with sparse indices without
    # converting back to dense, so we'll use a hybrid approach.
    # Actually, let's stick to the dense matrix approach but optimize the loop.
    
    # Alternative: Use a priority queue or just iterate.
    # With 50k blocks and 4k subsets, a simple loop in Python is slow.
    # Let's try to compute gains for all blocks at once using matrix multiplication
    # if we have the dense matrix.
    
    try:
        coverage_matrix = np.zeros((len(all_blocks), num_subsets), dtype=np.int8)
        for i, indices in enumerate(block_indices):
            for idx in indices:
                coverage_matrix[i, idx] = 1
        
        while uncovered.any():
            # Calculate gain: sum of coverage_matrix[i] * uncovered[i]
            # uncovered is bool, cast to int8 for multiplication
            uncovered_int = uncovered.astype(np.int8)
            gains = coverage_matrix @ uncovered_int
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            
            blocks.append(list(all_blocks[best_idx]))
            # Update uncovered
            uncovered[coverage_matrix[best_idx] == 1] = False
            
            if time.time() - start_time > time_limit:
                break
    except MemoryError:
        # Fallback: simple greedy picking best block from candidates containing first uncovered
        while uncovered.any():
            first_idx = np.where(uncovered)[0][0]
            first_subset = all_subsets[first_idx]
            
            best_block = None
            best_gain = -1
            remaining_points = [x for x in range(v) if x not in first_subset]
            
            for candidate_extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(list(first_subset) + list(candidate_extra)))
                gain = 0
                for combo in itertools.combinations(block, t):
                    idx = t_index_map[combo]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block:
                blocks.append(list(best_block))
                for combo in itertools.combinations(best_block, t):
                    idx = t_index_map[combo]
                    uncovered[idx] = False

    return blocks

def _repair(v, k, t, blocks):
    """Repair and reduce a list of blocks."""
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=5.0)
        
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    
    # Compute coverage mask for current blocks
    total_mask = np.zeros(num_subsets, dtype=bool)
    for b in blocks:
        for combo in itertools.combinations(sorted(b), t):
            idx = t_index_map[combo]
            total_mask[idx] = True
            
    uncovered = ~total_mask
    
    # Add blocks to cover remaining
    if uncovered.any():
        # Quick greedy add
        while uncovered.any():
            first_idx = np.where(uncovered)[0][0]
            first_subset = all_subsets[first_idx]
            
            best_block = None
            best_gain = -1
            
            remaining_points = [x for x in range(v) if x not in first_subset]
            for candidate_extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(list(first_subset) + list(candidate_extra)))
                gain = 0
                for combo in itertools.combinations(block, t):
                    idx = t_index_map[combo]
                    if uncovered[idx]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_block = block
            
            if best_block:
                blocks.append(list(best_block))
                for combo in itertools.combinations(best_block, t):
                    idx = t_index_map[combo]
                    uncovered[idx] = False

    # Reduction Phase: Remove redundant blocks
    n_blocks = len(blocks)
    if n_blocks <= 1:
        return blocks
        
    block_masks = []
    for b in blocks:
        mask = np.zeros(num_subsets, dtype=bool)
        for combo in itertools.combinations(sorted(b), t):
            idx = t_index_map[combo]
            mask[idx] = True
        block_masks.append(mask)
        
    keep = [True] * n_blocks
    for i in range(n_blocks):
        if not keep[i]:
            continue
        
        # Check if block i is redundant: union of others covers it
        # total_mask_without_i = total_mask & ~block_masks[i]
        # If total_mask_without_i == total_mask, then i is redundant.
        
        total_mask_without_i = total_mask & ~block_masks[i]
        if np.all(total_mask_without_i == total_mask):
            keep[i] = False
            
    return [b for b, k_flag in zip(blocks, keep) if k_flag]

def _crossover(parent1, parent2, v, k, t):
    """Merge two coverings using greedy selection from union."""
    # Union of blocks, preserving uniqueness
    seen = set()
    pool = []
    for b in parent1 + parent2:
        tb = tuple(sorted(b))
        if tb not in seen:
            seen.add(tb)
            pool.append(tb)
    
    # Greedy selection from pool
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    uncovered = np.ones(num_subsets, dtype=bool)
    child_blocks = []
    
    # Pre-compute masks for pool
    pool_masks = []
    for b in pool:
        mask = np.zeros(num_subsets, dtype=bool)
        for combo in itertools.combinations(b, t):
            idx = t_index_map[combo]
            mask[idx] = True
        pool_masks.append(mask)
        
    available = list(range(len(pool)))
    
    while uncovered.any() and available:
        best_idx = -1
        best_gain = -1
        
        for i in available:
            gain = np.sum(uncovered & pool_masks[i])
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                
        if best_idx == -1 or best_gain == 0:
            break
            
        child_blocks.append(list(pool[best_idx]))
        uncovered[pool_masks[best_idx]] = False
        available.remove(best_idx)
        
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = _repair(v, k, t, child_blocks)
        
    return child_blocks

def _local_search(v, k, t, blocks, time_limit=5.0):
    """Local search: try to swap blocks to reduce count."""
    start_time = time.time()
    current_blocks = [list(b) for b in blocks]
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    
    # Pre-compute masks for current blocks
    def get_mask(block):
        mask = np.zeros(num_subsets, dtype=bool)
        for combo in itertools.combinations(sorted(block), t):
            idx = t_index_map[combo]
            mask[idx] = True
        return mask
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(len(current_blocks)):
            # Try removing block i
            mask_i = get_mask(current_blocks[i])
            # Check if it's redundant
            # We need the union of all other masks
            # This is expensive if done naively.
            # Approximation: check if any single other block covers it? No.
            # Let's just try swapping block i with a random block from the universe.
            
            # Actually, a better local search:
            # 1. Remove a random block.
            # 2. Check if coverage is lost.
            # 3. If lost, try to add a different block that covers the lost parts better.
            
            # Simpler: Try to replace block i with a random block j.
            # If the new set is still a valid cover and has same size, accept?
            # We want to reduce size, so replacement doesn't help directly unless
            # it allows subsequent removals.
            
            # Let's stick to the repair function which does reduction.
            # Local search here could be:
            # For each block i, try to find a block j not in the set such that
            # replacing i with j allows removal of another block k.
            
            # This is complex. Let's just rely on the GA crossover and repair.
            # But we can do a simple "shake":
            # Randomly remove a block, then repair.
            # If the repaired result is smaller, accept.
            
            if len(current_blocks) > 1:
                idx_to_remove = random.randint(0, len(current_blocks) - 1)
                candidate = current_blocks[:idx_to_remove] + current_blocks[idx_to_remove+1:]
                candidate = _repair(v, k, t, candidate)
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    improved = True
                    break
        
        if not improved:
            # No improvement found in this pass, try a few more random perturbations
            for _ in range(5):
                if not current_blocks:
                    break
                idx_to_remove = random.randint(0, len(current_blocks) - 1)
                candidate = current_blocks[:idx_to_remove] + current_blocks[idx_to_remove+1:]
                candidate = _repair(v, k, t, candidate)
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    improved = True
                    break
            
            if not improved:
                break
                
    return current_blocks

def construct(v, k, t):
    """Main entry point."""
    start_time = time.time()
    time_limit = 115.0
    
    # Initial population
    # 1. Greedy
    greedy1 = _greedy_cover(v, k, t, time_limit=10.0)
    # 2. Random + Repair
    random_blocks = [sorted(random.sample(range(v), k)) for _ in range(100)]
    random1 = _repair(v, k, t, random_blocks)
    
    population = [greedy1, random1]
    
    # Add a few more diverse individuals
    for _ in range(4):
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(50)]
        pop_ind = _repair(v, k, t, rand_blocks)
        population.append(pop_ind)
        
    best_individual = min(population, key=len)
    
    while time.time() - start_time < time_limit:
        # Sort
        population.sort(key=len)
        
        # Elitism
        elite = population[:max(2, len(population)//2)]
        
        new_population = list(elite)
        
        # Generate children
        while len(new_population) < len(population):
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t)
            
            # Mutation: Replace a random block with a random one, then repair
            if random.random() < 0.5 and child:
                idx = random.randint(0, len(child)-1)
                child[idx] = sorted(random.sample(range(v), k))
                child = _repair(v, k, t, child)
            
            # Local search on the child
            if time.time() - start_time < time_limit - 10:
                child = _local_search(v, k, t, child, time_limit=2.0)
            
            new_population.append(child)
            
        population = new_population
        current_best = min(population, key=len)
        if len(current_best) < len(best_individual):
            best_individual = current_best
            if len(best_individual) <= 151:
                break
                
    # Final repair
    final_blocks = _repair(v, k, t, best_individual)
    
    # Validate
    all_subsets, _ = _get_all_t_subsets(v, t)
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
            
    if len(covered) != len(all_subsets):
        # Fallback
        final_blocks = _greedy_cover(v, k, t, time_limit=110.0)
        
    return final_blocks
# EVOLVE-BLOCK-END
