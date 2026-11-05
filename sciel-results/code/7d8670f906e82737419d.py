"""Genetic algorithm covering design C(19,7,4) with optimized crossover and repair."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

# Global cache for precomputed data
_CACHE = {}

def _get_precomputed(v, k, t):
    """Get or compute precomputed data for the given parameters."""
    key = (v, k, t)
    if key in _CACHE:
        return _CACHE[key]
    
    start_time = time.time()
    all_subsets = list(itertools.combinations(range(v), t))
    num_subsets = len(all_subsets)
    t_index_map = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Map block tuple to index for O(1) lookup
    block_index_map = {b: i for i, b in enumerate(all_blocks)}
    
    # Build coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.int8)
    # Also store the list of subset indices covered by each block for fast counting
    block_subsets = [list(itertools.combinations(b, t)) for b in all_blocks]
    block_subset_indices = []
    for i, block in enumerate(all_blocks):
        indices = []
        for combo in block_subsets[i]:
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
            indices.append(idx)
        block_subset_indices.append(indices)
        
    # Store block tuples for easy access
    block_tuples = [tuple(b) for b in all_blocks]
    
    data = {
        'all_subsets': all_subsets,
        't_index_map': t_index_map,
        'num_subsets': num_subsets,
        'all_blocks': all_blocks,
        'block_tuples': block_tuples,
        'coverage_matrix': coverage_matrix,
        'num_blocks': num_blocks,
        'block_index_map': block_index_map,
        'block_subset_indices': block_subset_indices,
    }
    
    _CACHE[key] = data
    return data

def _block_to_block_idx(block, v, k, t):
    """Convert a block (list/tuple of sorted ints) to its index in all_blocks."""
    data = _get_precomputed(v, k, t)
    block_set = frozenset(block)
    # Since all_blocks are generated from combinations, they are sorted tuples.
    # We can just convert to tuple and look up.
    # Note: input block might be a list, so convert to tuple.
    b_tuple = tuple(sorted(block))
    return data['block_index_map'].get(b_tuple, -1)

def _greedy_cover(v, k, t, time_limit=15.0):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    uncovered = np.ones(num_subsets, dtype=np.int8)
    blocks = []
    
    while uncovered.any():
        gains = coverage_matrix @ uncovered
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        
        blocks.append(list(data['block_tuples'][best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        
        if time.time() - start_time > time_limit:
            break
    
    return blocks

def _repair(v, k, t, blocks, time_limit=10.0):
    """Repair a list of blocks to ensure full coverage, then reduce."""
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    block_subset_indices = data['block_subset_indices']
    
    # 1. Normalize and deduplicate blocks
    unique_blocks = []
    seen_indices = set()
    for b in blocks:
        bi = _block_to_block_idx(b, v, k, t)
        if bi >= 0 and bi not in seen_indices:
            seen_indices.add(bi)
            unique_blocks.append(bi)
            
    if not unique_blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
        
    # 2. Check coverage and add missing blocks if necessary
    # Use a count array to track coverage of each 4-subset
    counts = np.zeros(num_subsets, dtype=np.int16)
    for bi in unique_blocks:
        for idx in block_subset_indices[bi]:
            counts[idx] += 1
            
    uncovered_mask = (counts == 0)
    
    if uncovered_mask.any():
        uncovered_int = uncovered_mask.astype(np.int8)
        while uncovered_int.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered_int
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            unique_blocks.append(best_idx)
            for idx in block_subset_indices[best_idx]:
                counts[idx] += 1
            uncovered_int = (counts == 0).astype(np.int8)
            
    # 3. Reduction phase: iteratively remove redundant blocks
    # We use a greedy removal strategy: try to remove each block if possible.
    # To be efficient, we iterate until no more blocks can be removed.
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_unique_blocks = []
        for bi in unique_blocks:
            # Tentatively remove bi
            for idx in block_subset_indices[bi]:
                counts[idx] -= 1
                
            # Check if any count dropped to 0
            if (counts == 0).any():
                # Restore bi
                for idx in block_subset_indices[bi]:
                    counts[idx] += 1
                new_unique_blocks.append(bi)
            else:
                # bi was redundant, keep it removed
                changed = True
        
        unique_blocks = new_unique_blocks
        
        if len(unique_blocks) <= 1:
            break
    
    # Reconstruct blocks from indices
    result = [list(data['block_tuples'][bi]) for bi in unique_blocks]
    return result

def _crossover(parent1, parent2, v, k, t, time_limit=10.0):
    """Custom crossover: merge two coverings by greedy selection from union."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Build union of blocks from both parents
    seen = set()
    pool_indices = []
    for b in parent1 + parent2:
        bi = _block_to_block_idx(b, v, k, t)
        if bi >= 0 and bi not in seen:
            seen.add(bi)
            pool_indices.append(bi)
    
    # Greedy selection from pool
    uncovered = np.ones(num_subsets, dtype=np.int8)
    child_blocks = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for all available blocks
        # Vectorized: gains = coverage_matrix[available] @ uncovered
        # But available changes size, so we do it in a loop or slice.
        # Since pool is small (max ~300), loop is fine.
        best_idx = -1
        best_gain = -1
        best_pos = -1
        
        for pos, bi in enumerate(available):
            gain = int(np.sum(coverage_matrix[bi] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_idx = bi
                best_pos = pos
        
        if best_idx == -1 or best_gain == 0:
            break
        
        child_blocks.append(list(data['block_tuples'][best_idx]))
        uncovered[coverage_matrix[best_idx] == 1] = 0
        available.pop(best_pos)
    
    # If still uncovered, repair
    if uncovered.any():
        child_blocks = _repair(v, k, t, child_blocks, time_limit=time_limit)
    
    return child_blocks

def _local_search(v, k, t, blocks, time_limit=8.0):
    """Local search: try removing blocks and repairing to find smaller coverings."""
    start_time = time.time()
    current_blocks = [list(b) for b in blocks]
    
    while time.time() - start_time < time_limit:
        improved = False
        
        # Try removing each block and see if repair yields a smaller set
        # We optimize by checking if removal is valid first, then only repairing if it breaks coverage
        # But _repair is now fast enough (O(N^2 * t) approx) that we can just call it.
        # However, we can be smarter: if we remove a block and coverage is still full, we don't need to add.
        
        for i in range(len(current_blocks)):
            if time.time() - start_time >= time_limit:
                break
            
            candidate = current_blocks[:i] + current_blocks[i+1:]
            
            # Quick check: does candidate cover everything?
            # We can use _repair which handles this. 
            # To save time, we can just call _repair.
            candidate = _repair(v, k, t, candidate, time_limit=2.0)
            
            if len(candidate) < len(current_blocks):
                current_blocks = candidate
                improved = True
                break
        
        if not improved:
            # Try random perturbations
            for _ in range(3):
                if time.time() - start_time >= time_limit or not current_blocks:
                    break
                
                idx_to_remove = random.randint(0, len(current_blocks) - 1)
                candidate = current_blocks[:idx_to_remove] + current_blocks[idx_to_remove+1:]
                candidate = _repair(v, k, t, candidate, time_limit=2.0)
                
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    improved = True
                    break
            
            if not improved:
                break
    
    return current_blocks

def construct(v, k, t):
    """Main entry point: genetic algorithm with custom crossover."""
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute
    data = _get_precomputed(v, k, t)
    
    # Initial population: diverse starting points
    population = []
    
    # 1. Standard greedy
    greedy1 = _greedy_cover(v, k, t, time_limit=15.0)
    population.append(greedy1)
    
    # 2. Greedy with random tie-breaking (run multiple times)
    for _ in range(3):
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0)
        # Add some randomness by shuffling equal-gain blocks
        if greedy_r:
            # Perturb: replace a few blocks with random ones, then repair
            perturbed = [list(b) for b in greedy_r]
            num_perturb = max(1, len(perturbed) // 10)
            for _ in range(num_perturb):
                idx = random.randint(0, len(perturbed) - 1)
                perturbed[idx] = sorted(random.sample(range(v), k))
            perturbed = _repair(v, k, t, perturbed, time_limit=10.0)
            population.append(perturbed)
    
    # 3. Random + repair
    for _ in range(3):
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(80)]
        rand_repaired = _repair(v, k, t, rand_blocks, time_limit=10.0)
        population.append(rand_repaired)
    
    best_individual = min(population, key=len)
    
    # Genetic algorithm loop
    pop_size = 8
    population = population[:pop_size]
    
    while time.time() - start_time < time_limit - 15:
        # Sort by size
        population.sort(key=len)
        
        # Elitism: keep top half
        elite_count = max(2, len(population) // 2)
        elite = population[:elite_count]
        
        new_population = list(elite)
        
        # Generate children via crossover
        while len(new_population) < pop_size:
            if time.time() - start_time >= time_limit - 10:
                break
            
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover(p1, p2, v, k, t, time_limit=8.0)
            
            # Mutation: occasionally replace a block with a random one
            if random.random() < 0.3 and child:
                idx = random.randint(0, len(child) - 1)
                child[idx] = sorted(random.sample(range(v), k))
                child = _repair(v, k, t, child, time_limit=8.0)
            
            # Local search
            if time.time() - start_time < time_limit - 10:
                child = _local_search(v, k, t, child, time_limit=5.0)
            
            new_population.append(child)
        
        population = new_population
        current_best = min(population, key=len)
        
        if len(current_best) < len(best_individual):
            best_individual = current_best
            if len(best_individual) <= 151:
                break
    
    # Final repair and validation
    final_blocks = _repair(v, k, t, best_individual, time_limit=10.0)
    
    # Validate coverage
    all_subsets = data['all_subsets']
    covered = set()
    for b in final_blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered.add(combo)
    
    if len(covered) != len(all_subsets):
        # Fallback to greedy
        final_blocks = _greedy_cover(v, k, t, time_limit=110.0)
    
    return final_blocks
# EVOLVE-BLOCK-END
