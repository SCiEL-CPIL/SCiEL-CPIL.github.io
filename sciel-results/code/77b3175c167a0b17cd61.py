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

def _block_to_mask(block, t_index_map):
    """Convert a block to a boolean mask of covered t-subsets."""
    mask = np.zeros(len(t_index_map), dtype=bool)
    for combo in itertools.combinations(sorted(block), 4): # t is always 4 for this task, but let's keep it generic if needed. 
        # Actually, the task fixes t=4, v=19, k=7. 
        # To be safe and fast, I will hardcode t=4 logic in the hot loops if I can, 
        # but the function signature takes t. 
        # C(7,4) = 35 combinations per block.
        idx = t_index_map.get(combo)
        if idx is not None:
            mask[idx] = True
    return mask

def _greedy_cover(v, k, t, time_limit=10.0):
    """Fast greedy covering. Returns list of blocks."""
    start_time = time.time()
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    uncovered = np.ones(num_subsets, dtype=bool)
    blocks = []
    
    # Pre-compute all possible blocks and their masks? 
    # C(19,7) = 50388. 
    # 50388 * 35 = ~1.7M operations to pre-compute masks.
    # This is feasible and will make the greedy step very fast (just max over masks).
    
    # However, memory for 50388 masks of length 3876 (bool) is 50388 * 3876 bytes ~ 195 MB.
    # This might be too much or slow to allocate.
    # Alternative: On-the-fly calculation but optimized.
    
    # Let's try on-the-fly but with a better heuristic.
    # Actually, for v=19, k=7, t=4, we can pre-compute the "coverage vector" for each possible block.
    # Let's check memory: 50388 blocks * 3876 bits = 50388 * 484.5 bytes = ~24 MB. 
    # If we use uint8 or bool, it's larger. 
    # Let's use a list of sets of indices for the greedy step? 
    # No, numpy boolean arrays are faster for 'any' and 'sum'.
    
    # Let's stick to the current approach but optimize the inner loop.
    # The current greedy is slow because it builds the block point by point.
    # A standard greedy set cover picks the block with the max coverage among ALL possible blocks.
    # Since we can generate all 50388 blocks, we can pre-compute their coverage.
    
    all_blocks = list(itertools.combinations(range(v), k))
    # Map block tuple to index for quick lookup?
    # We just need the masks.
    
    # Pre-computing masks for all 50388 blocks takes ~1-2 seconds.
    # Then greedy is just:
    # while uncovered.any():
    #    find block with max sum(uncovered & mask)
    #    add to blocks
    #    uncovered &= ~mask
    
    # This is O(N_blocks * N_subsets) per step.
    # N_steps is ~150.
    # 150 * 50388 * 3876 is too slow in Python loops.
    # We need vectorized operations.
    
    # Let's pre-compute a matrix? 50388 x 3876 is 195M booleans. 
    # numpy can handle this if we use uint8 or bool.
    # 195MB is acceptable.
    
    # Let's try to pre-compute the coverage matrix.
    # If it's too slow, we fallback to the point-by-point greedy.
    
    try:
        coverage_matrix = np.zeros((len(all_blocks), num_subsets), dtype=bool)
        for i, block in enumerate(all_blocks):
            for combo in itertools.combinations(block, t):
                idx = t_index_map[combo]
                coverage_matrix[i, idx] = True
        
        # Greedy Set Cover
        while uncovered.any():
            # Calculate gain for each block
            # gain = sum(uncovered & coverage_matrix[i])
            # This is 50388 dot products of length 3876.
            # Can be done with matrix multiplication?
            # uncovered is 1D. coverage_matrix is 2D.
            # gains = coverage_matrix @ uncovered (if uncovered is float)
            uncovered_float = uncovered.astype(np.float32)
            gains = coverage_matrix @ uncovered_float
            
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break # Should not happen if uncovered.any()
            
            blocks.append(list(all_blocks[best_idx]))
            uncovered[coverage_matrix[best_idx]] = False
            
            if time.time() - start_time > time_limit:
                break
    except MemoryError:
        # Fallback to slower greedy if memory fails
        pass
        
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
        # Use a fast greedy to add blocks
        # We can reuse the greedy logic but starting from current state
        # For simplicity, just run a quick greedy on the uncovered part
        # Note: _greedy_cover returns a full cover. We need to merge.
        # Let's just generate a greedy cover for the whole thing and merge?
        # No, that defeats the purpose.
        
        # Quick greedy add:
        # Pick first uncovered, build block.
        while uncovered.any():
            first_idx = np.where(uncovered)[0][0]
            first_subset = all_subsets[first_idx]
            
            # Find best block containing first_subset
            # Candidates: all k-subsets containing first_subset
            # C(19-4, 7-4) = C(15, 3) = 455 candidates.
            # This is small!
            
            best_block = None
            best_gain = -1
            
            remaining_points = [x for x in range(v) if x not in first_subset]
            for candidate_extra in itertools.combinations(remaining_points, k - t):
                block = tuple(sorted(list(first_subset) + list(candidate_extra)))
                # Calculate gain
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
    # A block is redundant if its coverage is a subset of the union of all OTHER blocks.
    # This is O(N^2) in the number of blocks. N~150. 150^2 = 22500 checks.
    # Each check is O(C(k,t)) = 35 operations. Fast enough.
    
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
        # Check if block i is redundant
        # Union of all other blocks
        # Instead of unioning all others, check if (mask_i & ~union_others) is empty
        # Or simpler: 
        # For each j != i, if mask_i is subset of mask_j, i is redundant. (Strong redundancy)
        # Weaker: union of others covers i.
        
        # Let's try strong redundancy first (faster)
        is_redundant = False
        for j in range(n_blocks):
            if i == j:
                continue
            if not keep[j]:
                continue
            # Check if mask_i is subset of mask_j
            # np.all(mask_i[~mask_j] == False)
            if np.all(block_masks[i][~block_masks[j]] == False):
                is_redundant = True
                break
                
        if is_redundant:
            keep[i] = False
        else:
            # Check if union of others covers i
            # This is expensive. Let's skip for now and rely on strong redundancy + greedy order.
            # Actually, let's do a simpler check: 
            # If we remove i, does the coverage change?
            # We can compute total_mask_without_i = total_mask & ~block_masks[i]
            # If total_mask_without_i == total_mask, then i is redundant.
            # This is O(1) numpy ops per block!
            
            total_mask_without_i = total_mask & ~block_masks[i]
            if np.all(total_mask_without_i == total_mask):
                keep[i] = False
                
    return [b for b, k_flag in zip(blocks, keep) if k_flag]

def _crossover(parent1, parent2, v, k, t):
    """Merge two coverings."""
    # Union of blocks
    pool = list(set([tuple(b) for b in parent1 + parent2]))
    
    # Greedy selection from pool
    all_subsets, t_index_map = _get_all_t_subsets(v, t)
    num_subsets = len(all_subsets)
    uncovered = np.ones(num_subsets, dtype=bool)
    child_blocks = []
    
    # Pre-compute masks for pool? Pool size is small (< 300).
    pool_masks = []
    for b in pool:
        mask = np.zeros(num_subsets, dtype=bool)
        for combo in itertools.combinations(sorted(b), t):
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
            
        child_blocks.append(pool[best_idx])
        uncovered[pool_masks[best_idx]] = False
        available.remove(best_idx)
        
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = _repair(v, k, t, child_blocks)
        
    return child_blocks

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
    # Simple check
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
