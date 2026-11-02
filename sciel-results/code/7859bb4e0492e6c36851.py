"""
Strategy: Simulated Annealing over valid coverings with bitmask-accelerated coverage tracking.
- Build an initial valid covering using a fast greedy.
- Generate a pool of candidate blocks.
- Use simulated annealing to minimize block count:
  * Move types: remove a redundant block, swap one block for another.
  * Track coverage counts per triple to quickly assess validity.
"""

import itertools
import random
import time
import math

def _greedy_build_cover(v, k, t, rng=None):
    """Build a valid covering using greedy set cover with randomized tie-breaking."""
    if rng is None:
        rng = random.Random(0)
    
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    uncovered_count = [1] * num_triples
    uncovered_set = set(range(num_triples))
    blocks = []
    
    while uncovered_set:
        # Greedily pick the block that covers the most uncovered triples
        best_block = None
        best_gain = -1
        best_block_triples = None
        
        # Start from a random uncovered triple to create a block
        start_triple = tuple(uncovered_set.pop())
        remaining_uncovered = set(uncovered_set)
        block = list(start_triple)
        covered_in_block = {start_triple}
        
        # Add remaining points greedily
        for _ in range(k - t):
            best_point = None
            best_local_gain = -1
            for x in range(v):
                if x in block:
                    continue
                # Count how many new uncovered triples this point would cover
                gain = 0
                for subset in itertools.combinations(block, t - 1):
                    new_triple = tuple(sorted(subset + (x,)))
                    if new_triple in remaining_uncovered or new_triple == start_triple:
                        if new_triple in remaining_uncovered:
                            gain += 1
                # Also count if it covers the start_triple (shouldn't be possible since we started with it)
                if gain > best_local_gain:
                    best_local_gain = gain
                    best_point = x
            if best_point is None:
                break
            block.append(best_point)
            # Update remaining_uncovered
            for subset in itertools.combinations(block, t):
                remaining_uncovered.discard(subset)
        
        block.sort()
        blocks.append(block)
        
        # Update uncovered
        for triple in itertools.combinations(block, t):
            uncovered_set.discard(triple)
    
    return blocks

def _generate_block_pool(v, k, t, num_pool, seed, greedy_blocks):
    """Generate a pool of candidate blocks including greedy blocks and random perturbations."""
    rng = random.Random(seed)
    pool = set()
    
    # Add greedy blocks
    for block in greedy_blocks:
        pool.add(tuple(block))
    
    # Add random blocks with some structure
    while len(pool) < num_pool:
        # Create a random block with some bias towards covering triples
        block = list(rng.sample(range(v), k))
        pool.add(tuple(sorted(block)))
        
    return list(pool)

def _compute_block_triples(block, t, triple_to_idx):
    """Return list of triple indices covered by a block."""
    triples = []
    for triple in itertools.combinations(block, t):
        idx = triple_to_idx.get(tuple(triple))
        if idx is not None:
            triples.append(idx)
    return triples

def _is_valid_cover(selected_blocks, block_triples_list, num_triples):
    """Check if the selected blocks form a valid cover."""
    coverage = [0] * num_triples
    for idx in selected_blocks:
        for triple_idx in block_triples_list[idx]:
            coverage[triple_idx] += 1
    return all(c > 0 for c in coverage)

def _count_uncovered(selected_blocks, block_triples_list, num_triples):
    """Count number of uncovered triples."""
    coverage = [0] * num_triples
    for idx in selected_blocks:
        for triple_idx in block_triples_list[idx]:
            coverage[triple_idx] += 1
    uncovered = sum(1 for c in coverage if c == 0)
    return uncovered

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Step 1: Build initial greedy cover
    rng0 = random.Random(42)
    initial_blocks = _greedy_build_cover(v, k, t, rng0)
    
    # Step 2: Generate block pool
    num_pool = min(500, 100 + 30 * len(initial_blocks))
    pool = _generate_block_pool(v, k, t, num_pool, seed=123, greedy_blocks=initial_blocks)
    
    # Step 3: Precompute triple indices for each pool block
    all_triples = list(itertools.combinations(range(v), t))
    triple_to_idx = {triple: i for i, triple in enumerate(all_triples)}
    num_triples = len(all_triples)
    
    block_triples_list = []
    for block in pool:
        block_triples_list.append(_compute_block_triples(block, t, triple_to_idx))
    
    # Map from greedy block to pool index
    block_to_pool_idx = {}
    for i, block in enumerate(pool):
        block_to_pool_idx[block] = i
    
    # Initial selected blocks as pool indices
    selected = []
    for block in initial_blocks:
        if tuple(block) in block_to_pool_idx:
            selected.append(block_to_pool_idx[tuple(block)])
    
    # If some initial blocks aren't in pool, add them
    for block in initial_blocks:
        tb = tuple(block)
        if tb not in block_to_pool_idx:
            pool.append(tb)
            block_to_pool_idx[tb] = len(pool) - 1
            block_triples_list.append(_compute_block_triples(tb, t, triple_to_idx))
            selected.append(block_to_pool_idx[tb])
    
    # Ensure we have a valid cover
    if not _is_valid_cover(selected, block_triples_list, num_triples):
        # Fallback: just return the greedy blocks
        return initial_blocks
    
    best_selected = list(selected)
    best_count = len(best_selected)
    
    # Step 4: Simulated Annealing
    rng = random.Random(99)
    temperature = 1.0
    cooling_rate = 0.9995
    min_temp = 0.01
    
    # Compute coverage for initial solution
    def _get_coverage(sel):
        cov = [0] * num_triples
        for idx in sel:
            for ti in block_triples_list[idx]:
                cov[ti] += 1
        return cov
    
    current_cov = _get_coverage(selected)
    current_uncovered = sum(1 for c in current_cov if c == 0)
    
    sa_iter = 0
    while time.time() < deadline - 5.0 and temperature > min_temp:
        sa_iter += 1
        
        # Choose move type
        move_type = rng.random()
        
        if move_type < 0.3 and len(selected) > 1:
            # Remove a block
            idx_to_remove = rng.choice(selected)
            # Check if removal keeps coverage valid
            new_cov = [0] * num_triples
            for idx in selected:
                if idx != idx_to_remove:
                    for ti in block_triples_list[idx]:
                        new_cov[ti] += 1
            new_uncovered = sum(1 for c in new_cov if c == 0)
            
            if new_uncovered == 0:
                # Valid improvement
                selected.remove(idx_to_remove)
                current_cov = new_cov
                current_uncovered = 0
                if len(selected) < best_count:
                    best_count = len(selected)
                    best_selected = list(selected)
            else:
                # Accept with probability
                delta = new_uncovered - current_uncovered
                if delta <= 0 or rng.random() < math.exp(-delta / temperature):
                    # Revert (we didn't apply)
                    pass
        elif move_type < 0.8:
            # Swap: remove one block, add another
            if len(selected) >= 1:
                idx_to_remove = rng.choice(selected)
                # Find a block to add that isn't already selected
                available = [i for i in range(len(pool)) if i not in set(selected)]
                if available:
                    idx_to_add = rng.choice(available)
                    
                    # Compute new coverage
                    new_cov = [0] * num_triples
                    for idx in selected:
                        if idx != idx_to_remove:
                            for ti in block_triples_list[idx]:
                                new_cov[ti] += 1
                    for ti in block_triples_list[idx_to_add]:
                        new_cov[ti] += 1
                    
                    new_uncovered = sum(1 for c in new_cov if c == 0)
                    
                    # Accept if it improves or with probability
                    delta_uncovered = new_uncovered - current_uncovered
                    delta_count = 0  # Swap keeps count same
                    if delta_uncovered < 0 or (delta_uncovered == 0 and rng.random() < 0.3):
                        # Apply swap
                        selected.remove(idx_to_remove)
                        selected.append(idx_to_add)
                        current_cov = new_cov
                        current_uncovered = new_uncovered
                    elif delta_uncovered <= 0 or rng.random() < math.exp(-delta_uncovered / temperature):
                        selected.remove(idx_to_remove)
                        selected.append(idx_to_add)
                        current_cov = new_cov
                        current_uncovered = new_uncovered
        else:
            # Add a block (rare, helps escape local minima)
            available = [i for i in range(len(pool)) if i not in set(selected)]
            if available:
                idx_to_add = rng.choice(available)
                new_cov = [0] * num_triples
                for idx in selected:
                    for ti in block_triples_list[idx]:
                        new_cov[ti] += 1
                for ti in block_triples_list[idx_to_add]:
                    new_cov[ti] += 1
                new_uncovered = sum(1 for c in new_cov if c == 0)
                
                delta = new_uncovered - current_uncovered
                # Adding a block increases count, so only accept if it significantly reduces uncovered
                if delta < -1 or (delta == -1 and rng.random() < 0.5):
                    selected.append(idx_to_add)
                    current_cov = new_cov
                    current_uncovered = new_uncovered
        
        temperature *= cooling_rate
    
    # Final local search: try to remove redundant blocks
    for _ in range(100):
        if time.time() > deadline - 2.0:
            break
        cov = [0] * num_triples
        for idx in best_selected:
            for ti in block_triples_list[idx]:
                cov[ti] += 1
        
        # Find removable blocks
        for idx in list(best_selected):
            # Check if this block is the only one covering any triple
            removable = True
            for ti in block_triples_list[idx]:
                if cov[ti] == 1:
                    removable = False
                    break
            if removable:
                best_selected.remove(idx)
                best_count = len(best_selected)
                break  # Restart scan after removal
    
    # Convert back to blocks
    final_blocks = [list(pool[idx]) for idx in best_selected]
    
    # Final validation
    all_triples_set = set(all_triples)
    covered = set()
    for block in final_blocks:
        for triple in itertools.combinations(block, t):
            covered.add(tuple(triple))
    
    if covered != all_triples_set:
        # Fallback to greedy
        return _greedy_build_cover(v, k, t, random.Random(0))
    
    return final_blocks
