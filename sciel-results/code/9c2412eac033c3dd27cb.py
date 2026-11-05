"""Genetic algorithm covering design C(19,7,4) with custom crossover."""
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
    
    # Precompute all k-blocks and their coverage masks
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build coverage matrix: coverage_matrix[i, j] = 1 if block i covers subset j
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.int8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
    
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
    }
    
    _CACHE[key] = data
    return data

def _block_to_block_idx(block, v, k, t):
    """Convert a block (list/tuple of sorted ints) to its index in all_blocks."""
    data = _get_precomputed(v, k, t)
    block_tuple = tuple(sorted(block))
    # Use the precomputed mapping if available
    if 'block_to_idx' not in data:
        data['block_to_idx'] = {bt: i for i, bt in enumerate(data['block_tuples'])}
    return data['block_to_idx'].get(block_tuple, -1)

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
    
    # Build current coverage mask and track block indices
    total_mask = np.zeros(num_subsets, dtype=np.int8)
    block_indices = []
    seen = set()
    for b in blocks:
        bi = _block_to_block_idx(b, v, k, t)
        if bi >= 0 and bi not in seen:
            seen.add(bi)
            block_indices.append(bi)
            total_mask[coverage_matrix[bi] == 1] = 1
    
    # Add blocks to cover remaining gaps
    uncovered_count = num_subsets - int(np.sum(total_mask))
    while uncovered_count > 0 and time.time() - start_time < time_limit:
        gains = coverage_matrix @ (1 - total_mask)
        best_idx = int(np.argmax(gains))
        if gains[best_idx] == 0:
            break
        
        block_indices.append(best_idx)
        total_mask[coverage_matrix[best_idx] == 1] = 1
        uncovered_count -= int(gains[best_idx])
    
    # Reduction phase: iteratively remove redundant blocks using count-based check
    if len(block_indices) <= 1:
        return [list(data['block_tuples'][bi]) for bi in block_indices]
    
    # Build coverage count array: how many blocks cover each subset
    coverage_count = np.zeros(num_subsets, dtype=np.int32)
    for bi in block_indices:
        coverage_count[coverage_matrix[bi] == 1] += 1
    
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_block_indices = []
        for i, bi in enumerate(block_indices):
            # Check if any subset would become uncovered if we remove this block
            block_mask = coverage_matrix[bi] == 1
            if np.any(coverage_count[block_mask] == 1):
                # This block is essential for at least one subset
                new_block_indices.append(bi)
            else:
                # This block is redundant - remove it
                coverage_count[coverage_matrix[bi] == 1] -= 1
                changed = True
        
        if changed:
            block_indices = new_block_indices
    
    # Reconstruct blocks from block_indices
    result = [list(data['block_tuples'][bi]) for bi in block_indices]
    
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
        for i in range(len(current_blocks)):
            if time.time() - start_time >= time_limit:
                break
            
            candidate = current_blocks[:i] + current_blocks[i+1:]
            candidate = _repair(v, k, t, candidate, time_limit=2.0)
            
            if len(candidate) < len(current_blocks):
                current_blocks = candidate
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one block, add a different one
            for _ in range(5):
                if time.time() - start_time >= time_limit or not current_blocks:
                    break
                
                idx_to_remove = random.randint(0, len(current_blocks) - 1)
                candidate = current_blocks[:idx_to_remove] + current_blocks[idx_to_remove+1:]
                # Add a random block
                candidate.append(sorted(random.sample(range(v), k)))
                candidate = _repair(v, k, t, candidate, time_limit=2.0)
                
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    improved = True
                    break
            
            if not improved:
                # Try multi-remove: remove 2 blocks at once
                if len(current_blocks) > 2:
                    i, j = random.sample(range(len(current_blocks)), 2)
                    candidate = [b for idx, b in enumerate(current_blocks) if idx not in (i, j)]
                    candidate = _repair(v, k, t, candidate, time_limit=3.0)
                    if len(candidate) < len(current_blocks):
                        current_blocks = candidate
                        improved = True
                    else:
                        break
                else:
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
    
    # 2. Greedy with perturbation (run multiple times)
    for _ in range(3):
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0)
        if greedy_r:
            # Perturb: replace a few blocks with random ones, then repair
            perturbed = [list(b) for b in greedy_r]
            num_perturb = max(1, len(perturbed) // 8)
            for _ in range(num_perturb):
                idx = random.randint(0, len(perturbed) - 1)
                perturbed[idx] = sorted(random.sample(range(v), k))
            perturbed = _repair(v, k, t, perturbed, time_limit=8.0)
            population.append(perturbed)
    
    # 3. Random + repair (more diverse starting points)
    for _ in range(2):
        rand_blocks = [sorted(random.sample(range(v), k)) for _ in range(60)]
        rand_repaired = _repair(v, k, t, rand_blocks, time_limit=8.0)
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
