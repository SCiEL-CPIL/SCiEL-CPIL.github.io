"""Genetic algorithm for covering design C(19,7,4) with union-greedy crossover."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

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
    
    all_blocks = list(itertools.combinations(range(v), k))
    num_blocks = len(all_blocks)
    
    # Build coverage matrix
    coverage_matrix = np.zeros((num_blocks, num_subsets), dtype=np.int8)
    for i, block in enumerate(all_blocks):
        for combo in itertools.combinations(block, t):
            idx = t_index_map[combo]
            coverage_matrix[i, idx] = 1
    
    block_tuples = [tuple(b) for b in all_blocks]
    
    # Map from frozenset to block index for fast lookup
    block_set_to_idx = {frozenset(b): i for i, b in enumerate(all_blocks)}
    
    data = {
        'all_subsets': all_subsets,
        't_index_map': t_index_map,
        'num_subsets': num_subsets,
        'all_blocks': all_blocks,
        'block_tuples': block_tuples,
        'coverage_matrix': coverage_matrix,
        'num_blocks': num_blocks,
        'block_set_to_idx': block_set_to_idx,
    }
    
    _CACHE[key] = data
    return data

def _blocks_to_indices(blocks, v, k, t):
    """Convert list of blocks to list of block indices."""
    data = _get_precomputed(v, k, t)
    lookup = data['block_set_to_idx']
    indices = []
    for b in blocks:
        idx = lookup.get(frozenset(b), -1)
        if idx >= 0:
            indices.append(idx)
    return indices

def _greedy_cover(v, k, t, time_limit=15.0, randomize=False):
    """Fast greedy covering using precomputed coverage matrix."""
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    num_blocks = data['num_blocks']
    
    uncovered = np.ones(num_subsets, dtype=np.int8)
    selected = []
    used = np.zeros(num_blocks, dtype=bool)
    
    while uncovered.any():
        if time.time() - start_time > time_limit:
            break
        
        gains = coverage_matrix @ uncovered
        gains[used] = 0
        
        if randomize:
            # Random tie-breaking among top candidates
            max_gain = int(np.max(gains))
            if max_gain == 0:
                break
            candidates = np.where(gains == max_gain)[0]
            best_idx = int(np.random.choice(candidates))
        else:
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
        
        selected.append(best_idx)
        used[best_idx] = True
        uncovered[coverage_matrix[best_idx] == 1] = 0
    
    return [list(data['block_tuples'][i]) for i in selected]

def _repair(v, k, t, blocks, time_limit=10.0):
    """Repair a list of blocks to ensure full coverage, then reduce."""
    if not blocks:
        return _greedy_cover(v, k, t, time_limit=time_limit)
    
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    num_blocks = data['num_blocks']
    
    block_indices = _blocks_to_indices(blocks, v, k, t)
    
    # Build current coverage mask
    total_mask = np.zeros(num_subsets, dtype=bool)
    for bi in block_indices:
        total_mask[coverage_matrix[bi] == 1] = True
    
    uncovered = ~total_mask
    
    # Add blocks to cover remaining gaps
    if uncovered.any():
        uncovered_int = uncovered.astype(np.int8)
        while uncovered.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered_int
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            block_indices.append(best_idx)
            total_mask[coverage_matrix[best_idx] == 1] = True
            uncovered = ~total_mask
            uncovered_int = uncovered.astype(np.int8)
    
    # Reduction phase: iteratively remove redundant blocks
    if len(block_indices) <= 1:
        return [list(data['block_tuples'][bi]) for bi in block_indices]
    
    changed = True
    while changed and time.time() - start_time < time_limit:
        changed = False
        new_indices = []
        for i, bi in enumerate(block_indices):
            if len(block_indices) <= 1:
                new_indices.append(bi)
                continue
            
            # Check if removing this block still covers everything
            mask_without = np.zeros(num_subsets, dtype=bool)
            for j, bj in enumerate(block_indices):
                if j != i:
                    mask_without[coverage_matrix[bj] == 1] = True
            
            if mask_without.all():
                changed = True
            else:
                new_indices.append(bi)
        
        if changed:
            block_indices = new_indices
    
    # Remove duplicates
    seen = set()
    unique_indices = []
    for bi in block_indices:
        if bi not in seen:
            seen.add(bi)
            unique_indices.append(bi)
    
    return [list(data['block_tuples'][bi]) for bi in unique_indices]

def _union_greedy_crossover(parent1, parent2, v, k, t, time_limit=10.0):
    """
    Custom crossover: merge two coverings by selecting blocks from the union
    that cover the most uncovered 4-subsets.
    """
    start_time = time.time()
    data = _get_precomputed(v, k, t)
    coverage_matrix = data['coverage_matrix']
    num_subsets = data['num_subsets']
    
    # Build union of blocks from both parents
    seen = set()
    pool_indices = []
    for b in parent1 + parent2:
        bi = data['block_set_to_idx'].get(frozenset(b), -1)
        if bi >= 0 and bi not in seen:
            seen.add(bi)
            pool_indices.append(bi)
    
    # Greedy selection from pool: pick block covering most uncovered subsets
    uncovered = np.ones(num_subsets, dtype=np.int8)
    child_indices = []
    available = list(pool_indices)
    
    while uncovered.any() and available and time.time() - start_time < time_limit:
        # Compute gains for all available blocks
        best_gain = -1
        best_pos = -1
        best_idx = -1
        
        for pos, bi in enumerate(available):
            gain = int(np.sum(coverage_matrix[bi] & uncovered))
            if gain > best_gain:
                best_gain = gain
                best_idx = bi
                best_pos = pos
        
        if best_idx == -1 or best_gain == 0:
            break
        
        child_indices.append(best_idx)
        uncovered[coverage_matrix[best_idx] == 1] = 0
        available.pop(best_pos)
    
    # If still uncovered, add greedy blocks from full space
    if uncovered.any():
        uncovered_int = uncovered.astype(np.int8)
        while uncovered.any() and time.time() - start_time < time_limit:
            gains = coverage_matrix @ uncovered_int
            best_idx = int(np.argmax(gains))
            if gains[best_idx] == 0:
                break
            
            child_indices.append(best_idx)
            uncovered[coverage_matrix[best_idx] == 1] = 0
            uncovered_int = uncovered.astype(np.int8)
    
    # Convert to blocks and repair
    child_blocks = [list(data['block_tuples'][bi]) for bi in child_indices]
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
            candidate = _repair(v, k, t, candidate, time_limit=3.0)
            
            if len(candidate) < len(current_blocks):
                current_blocks = candidate
                improved = True
                break
        
        if not improved:
            # Try random perturbations: swap a block with a random one
            for _ in range(5):
                if time.time() - start_time >= time_limit or not current_blocks:
                    break
                
                idx_to_replace = random.randint(0, len(current_blocks) - 1)
                candidate = current_blocks[:]
                candidate[idx_to_replace] = sorted(random.sample(range(v), k))
                candidate = _repair(v, k, t, candidate, time_limit=3.0)
                
                if len(candidate) < len(current_blocks):
                    current_blocks = candidate
                    improved = True
                    break
            
            if not improved:
                break
    
    return current_blocks

def construct(v, k, t):
    """Main entry point: genetic algorithm with union-greedy crossover."""
    start_time = time.time()
    time_limit = 115.0
    
    # Precompute
    data = _get_precomputed(v, k, t)
    
    # Initial population: diverse starting points
    population = []
    
    # 1. Standard greedy
    greedy1 = _greedy_cover(v, k, t, time_limit=15.0)
    population.append(greedy1)
    
    # 2. Greedy with random tie-breaking
    for _ in range(3):
        greedy_r = _greedy_cover(v, k, t, time_limit=10.0, randomize=True)
        population.append(greedy_r)
    
    # 3. Random + repair for diversity
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
        
        # Generate children via union-greedy crossover
        while len(new_population) < pop_size:
            if time.time() - start_time >= time_limit - 10:
                break
            
            # Select parents: one from elite, one from full population
            p1 = random.choice(elite)
            p2 = random.choice(population)
            
            child = _union_greedy_crossover(p1, p2, v, k, t, time_limit=8.0)
            
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
