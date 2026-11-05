"""Genetic algorithm covering design C(19,7,4)."""
# EVOLVE-BLOCK-START
import itertools
import random
import time
import numpy as np

def _all_t_subsets(v, t):
    """Generate all t-subsets of {0,...,v-1} as tuples."""
    return list(itertools.combinations(range(v), t))

def _precompute_coverage(v, k, t):
    """
    Pre-compute all k-subsets and the t-subsets they cover.
    Returns:
        all_k_blocks: List of all k-subsets (sorted tuples).
        block_to_idx: Map from k-subset tuple to index in all_k_blocks.
        covers: List of lists, where covers[i] is the list of indices of t-subsets covered by all_k_blocks[i].
        all_t_subs: List of all t-subsets.
        t_sub_to_idx: Map from t-subset tuple to index.
    """
    all_t_subs = _all_t_subsets(v, t)
    t_sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    block_to_idx = {b: i for i, b in enumerate(all_k_blocks)}
    
    covers = []
    for block in all_k_blocks:
        covered_indices = []
        for combo in itertools.combinations(block, t):
            idx = t_sub_to_idx.get(combo)
            if idx is not None:
                covered_indices.append(idx)
        covers.append(covered_indices)
        
    return all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx

def _greedy_cover_fast(v, k, t, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx, num_blocks=None):
    """Greedy covering using pre-computed data."""
    num_t_subs = len(all_t_subs)
    uncovered = np.ones(num_t_subs, dtype=bool)
    selected_blocks = []
    used_block_indices = set()
    
    max_blocks = num_blocks if num_blocks else num_t_subs
    
    # Pre-compute gain for each block? No, gain changes as uncovered changes.
    # We need to find the block that covers the most *currently uncovered* t-subsets.
    
    for _ in range(max_blocks):
        if not uncovered.any():
            break
        
        best_block_idx = -1
        best_gain = -1
        
        # Iterate through all possible blocks
        for b_idx, block in enumerate(all_k_blocks):
            if b_idx in used_block_indices:
                continue
            # Count how many uncovered t-subsets this block covers
            gain = 0
            for t_idx in covers[b_idx]:
                if uncovered[t_idx]:
                    gain += 1
            
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1 or best_gain == 0:
            break
            
        selected_blocks.append(list(all_k_blocks[best_block_idx]))
        used_block_indices.add(best_block_idx)
        
        # Update uncovered
        for t_idx in covers[best_block_idx]:
            uncovered[t_idx] = False
            
    return selected_blocks

def _crossover_fast(parent1, parent2, v, k, t, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx, max_blocks):
    """
    Crossover: Merge two coverings by selecting blocks that cover the most uncovered t-subsets.
    Parents are lists of blocks (lists of ints).
    """
    num_t_subs = len(all_t_subs)
    uncovered = np.ones(num_t_subs, dtype=bool)
    child_blocks = []
    
    # Pool of block indices from parents
    pool = []
    seen = set()
    for b in parent1:
        idx = block_to_idx.get(tuple(b))
        if idx is not None and idx not in seen:
            pool.append(idx)
            seen.add(idx)
    for b in parent2:
        idx = block_to_idx.get(tuple(b))
        if idx is not None and idx not in seen:
            pool.append(idx)
            seen.add(idx)
    
    if not pool:
        return []
        
    # Shuffle pool to break ties and introduce randomness
    random.shuffle(pool)
    
    for _ in range(max_blocks):
        if not uncovered.any():
            break
            
        best_idx_in_pool = -1
        best_gain = -1
        
        for i, b_idx in enumerate(pool):
            gain = 0
            for t_idx in covers[b_idx]:
                if uncovered[t_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_idx_in_pool = i
                
        if best_idx_in_pool == -1 or best_gain == 0:
            break
            
        selected_idx = pool[best_idx_in_pool]
        pool.pop(best_idx_in_pool)
        
        child_blocks.append(list(all_k_blocks[selected_idx]))
        
        for t_idx in covers[selected_idx]:
            uncovered[t_idx] = False
            
    return child_blocks

def _repair_fast(v, k, t, blocks, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx):
    """Repair a partial covering by greedily adding blocks to cover remaining t-subsets."""
    num_t_subs = len(all_t_subs)
    covered = np.zeros(num_t_subs, dtype=bool)
    
    # Mark covered subsets
    for b in blocks:
        idx = block_to_idx.get(tuple(b))
        if idx is not None:
            for t_idx in covers[idx]:
                covered[t_idx] = True
                
    uncovered = ~covered
    if not uncovered.any():
        return blocks
        
    blocks = list(blocks)
    used_indices = set()
    for b in blocks:
        idx = block_to_idx.get(tuple(b))
        if idx is not None:
            used_indices.add(idx)
            
    while uncovered.any():
        best_block_idx = -1
        best_gain = -1
        
        for b_idx in range(len(all_k_blocks)):
            if b_idx in used_indices:
                continue
            gain = 0
            for t_idx in covers[b_idx]:
                if uncovered[t_idx]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_block_idx = b_idx
                
        if best_block_idx == -1 or best_gain == 0:
            break
            
        blocks.append(list(all_k_blocks[best_block_idx]))
        used_indices.add(best_block_idx)
        
        for t_idx in covers[best_block_idx]:
            uncovered[t_idx] = False
            
    return blocks

def _remove_redundant_blocks(v, k, t, blocks, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx):
    """
    Remove blocks that do not cover any unique t-subset.
    A block is redundant if all t-subsets it covers are also covered by at least one other block.
    """
    if not blocks:
        return blocks
        
    num_blocks = len(blocks)
    # count[t_idx] = number of blocks that cover t_idx
    cover_counts = np.zeros(len(all_t_subs), dtype=int)
    
    block_indices = []
    for b in blocks:
        idx = block_to_idx.get(tuple(b))
        if idx is not None:
            block_indices.append(idx)
        else:
            # This shouldn't happen if blocks are valid
            block_indices.append(-1)
            
    for b_idx in block_indices:
        if b_idx != -1:
            for t_idx in covers[b_idx]:
                cover_counts[t_idx] += 1
                
    kept_blocks = []
    for i, b in enumerate(blocks):
        b_idx = block_indices[i]
        if b_idx == -1:
            continue
            
        is_redundant = True
        for t_idx in covers[b_idx]:
            if cover_counts[t_idx] == 1:
                is_redundant = False
                break
                
        if not is_redundant:
            kept_blocks.append(b)
            
    return kept_blocks

def _is_valid_covering(v, k, t, blocks, all_t_subs, t_sub_to_idx):
    """Check if the blocks form a valid covering."""
    num_t_subs = len(all_t_subs)
    covered = np.zeros(num_t_subs, dtype=bool)
    for b in blocks:
        # Use fast path if possible, but for validation we can just check
        # Since we have precomputed, let's use it
        # Note: _is_valid is called at the end, so speed is less critical than correctness
        pass
        
    # Simpler check: generate all covered 4-subsets and compare count
    covered_set = set()
    for b in blocks:
        for combo in itertools.combinations(sorted(b), t):
            covered_set.add(combo)
            
    return len(covered_set) == num_t_subs

def construct(v, k, t):
    """Main entry point: use genetic algorithm to find a covering design."""
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # Pre-compute
    all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx = _precompute_coverage(v, k, t)
    
    pop_size = 10
    max_blocks_per_individual = 200
    
    # Initialize population
    population = []
    
    # Add greedy solutions
    for i in range(4):
        blocks = _greedy_cover_fast(v, k, t, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx, num_blocks=max_blocks_per_individual)
        blocks = _remove_redundant_blocks(v, k, t, blocks, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
        population.append(blocks)
        
    # Add random + repaired
    for i in range(pop_size - 4):
        # Pick random blocks
        random_blocks = []
        for _ in range(50):
            b_idx = random.randint(0, len(all_k_blocks) - 1)
            random_blocks.append(list(all_k_blocks[b_idx]))
        blocks = _repair_fast(v, k, t, random_blocks, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
        blocks = _remove_redundant_blocks(v, k, t, blocks, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
        population.append(blocks)
        
    def fitness(blocks):
        return len(blocks)
        
    best_individual = min(population, key=fitness)
    best_fitness = fitness(best_individual)
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        population.sort(key=fitness)
        
        elite_size = max(1, pop_size // 2)
        elite = population[:elite_size]
        
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            p1 = random.choice(elite)
            p2 = random.choice(elite)
            
            child = _crossover_fast(p1, p2, v, k, t, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx, max_blocks_per_individual)
            
            # Repair
            child = _repair_fast(v, k, t, child, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
            
            # Remove redundant
            child = _remove_redundant_blocks(v, k, t, child, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
            
            # Mutation: replace a block with a random one
            if random.random() < 0.4 and child:
                idx = random.randint(0, len(child) - 1)
                new_b_idx = random.randint(0, len(all_k_blocks) - 1)
                child[idx] = list(all_k_blocks[new_b_idx])
                # Re-repair
                child = _repair_fast(v, k, t, child, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
                child = _remove_redundant_blocks(v, k, t, child, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
                
            new_population.append(child)
            
        population = new_population
        
        current_best = min(population, key=fitness)
        current_best_fitness = fitness(current_best)
        if current_best_fitness < best_fitness:
            best_individual = current_best
            best_fitness = current_best_fitness
            
        # Check if we've hit a good enough solution
        if best_fitness <= 151:
            break
            
    # Final cleanup
    final_blocks = _remove_redundant_blocks(v, k, t, best_individual, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
    
    # Verify validity
    if not _is_valid_covering(v, k, t, final_blocks, all_t_subs, t_sub_to_idx):
        # Fallback to pure greedy
        final_blocks = _greedy_cover_fast(v, k, t, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx, num_blocks=500)
        final_blocks = _remove_redundant_blocks(v, k, t, final_blocks, all_k_blocks, block_to_idx, covers, all_t_subs, t_sub_to_idx)
        
    return final_blocks
# EVOLVE-BLOCK-END
