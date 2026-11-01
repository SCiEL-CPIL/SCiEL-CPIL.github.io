"""
Genetic algorithm for covering design C(19,7,4).
Uses NumPy vectorized operations for fast fitness evaluation.
"""
import numpy as np
import time
import random
from itertools import combinations

# Precompute all t-subsets as indices
def _precompute_subsets(v, t):
    """Map each t-subset to an index, and store the subsets."""
    subsets = list(combinations(range(v), t))
    n_subsets = len(subsets)
    # Map from frozenset to index
    subset_to_idx = {frozenset(s): i for i, s in enumerate(subsets)}
    return subsets, subset_to_idx, n_subsets


def _subset_index(frozenset_subset, subset_to_idx):
    """Get index of a frozenset subset."""
    return subset_to_idx.get(frozenset_subset, -1)


def _build_block_matrix(block, v, k, t, subset_to_idx):
    """
    Build a boolean vector of length n_subsets indicating which t-subsets
    are covered by this block.
    """
    # For a block of size k, it covers all t-subsets that are contained in it
    # Number of t-subsets in a k-set is C(k, t)
    covered = np.zeros(1, dtype=bool)  # placeholder, we'll build properly
    # Actually, let's return a set of covered subset indices for this block
    covered_indices = set()
    for sub in combinations(block, t):
        idx = subset_to_idx.get(frozenset(sub), -1)
        if idx >= 0:
            covered_indices.add(idx)
    return covered_indices


def _fitness(blocks, v, k, t, subsets, subset_to_idx, n_subsets):
    """
    Compute fitness: number of uncovered t-subsets.
    Lower is better.
    """
    if not blocks:
        return n_subsets
    
    # Build coverage: for each t-subset, check if it's in any block
    # Use a boolean array of length n_subsets
    covered = np.zeros(n_subsets, dtype=bool)
    
    for block in blocks:
        for sub in combinations(block, t):
            idx = subset_to_idx.get(frozenset(sub), -1)
            if idx >= 0:
                covered[idx] = True
    
    uncovered = n_subsets - np.sum(covered)
    return uncovered


def _greedy_init(v, k, t, subsets, subset_to_idx, n_subsets, max_blocks=200):
    """
    Greedy initialization: start with random blocks, then add blocks
    to cover uncovered subsets.
    """
    blocks = []
    covered = np.zeros(n_subsets, dtype=bool)
    
    # Add some random blocks first for diversity
    n_random = min(10, max_blocks)
    for _ in range(n_random):
        block = sorted(random.sample(range(v), k))
        blocks.append(block)
        for sub in combinations(block, t):
            idx = subset_to_idx.get(frozenset(sub), -1)
            if idx >= 0:
                covered[idx] = True
    
    # Greedily add blocks to cover uncovered subsets
    uncovered_mask = ~covered
    while np.any(uncovered_mask) and len(blocks) < max_blocks:
        # Find an uncovered subset
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        # Pick a random uncovered subset
        idx = uncovered_indices[random.randint(0, len(uncovered_indices) - 1)]
        target_subset = subsets[idx]
        
        # Find the best point to add that covers the most uncovered subsets
        # Start with the target subset
        block = list(target_subset)
        
        # Greedily add points
        while len(block) < k:
            best_point = None
            best_gain = -1
            for p in range(v):
                if p in block:
                    continue
                # Count how many new uncovered subsets would be covered
                new_block = block + [p]
                gain = 0
                for sub in combinations(new_block, t):
                    idx2 = subset_to_idx.get(frozenset(sub), -1)
                    if idx2 >= 0 and not covered[idx2]:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_point = p
            if best_point is not None:
                block.append(best_point)
            else:
                break
        
        block.sort()
        blocks.append(block)
        
        # Update coverage
        for sub in combinations(block, t):
            idx2 = subset_to_idx.get(frozenset(sub), -1)
            if idx2 >= 0:
                covered[idx2] = True
                uncovered_mask[idx2] = False
    
    return blocks


def _crossover(parent1, parent2):
    """Uniform crossover between two block lists."""
    if not parent1 or not parent2:
        return parent1 or parent2
    
    # Combine all blocks and randomly select
    all_blocks = parent1 + parent2
    random.shuffle(all_blocks)
    # Keep a subset
    n_keep = max(len(parent1), len(parent2))
    return all_blocks[:n_keep]


def _mutate(block, v, k, t, mutation_rate=0.3):
    """
    Mutate a block: with some probability, replace a point.
    Returns a new block.
    """
    if random.random() > mutation_rate:
        return block
    
    new_block = block[:]
    # Replace one random point
    if len(new_block) == 0:
        return new_block
    idx = random.randint(0, len(new_block) - 1)
    old_point = new_block[idx]
    # Pick a new point not in the block
    candidates = [p for p in range(v) if p not in new_block]
    if candidates:
        new_point = random.choice(candidates)
        new_block[idx] = new_point
        new_block.sort()
    return new_block


def _add_random_block(v, k):
    """Add a random block."""
    return sorted(random.sample(range(v), k))


def _remove_random_block(blocks):
    """Remove a random block."""
    if not blocks:
        return blocks
    idx = random.randint(0, len(blocks) - 1)
    return blocks[:idx] + blocks[idx+1:]


def construct(v, k, t):
    """
    Genetic algorithm for covering design.
    """
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Precompute
    subsets, subset_to_idx, n_subsets = _precompute_subsets(v, t)
    
    # Initialize with greedy
    best_blocks = _greedy_init(v, k, t, subsets, subset_to_idx, n_subsets, max_blocks=180)
    best_uncovered = _fitness(best_blocks, v, k, t, subsets, subset_to_idx, n_subsets)
    
    # Population for GA
    pop_size = 30
    population = [best_blocks[:]]
    for _ in range(pop_size - 1):
        # Create variants of best
        variant = best_blocks[:]
        random.shuffle(variant)
        # Mutate some blocks
        for i in range(len(variant)):
            if random.random() < 0.2:
                variant[i] = _mutate(variant[i], v, k, t, mutation_rate=0.5)
        # Add/remove some blocks
        if random.random() < 0.3 and len(variant) > 50:
            variant = _remove_random_block(variant)
        if random.random() < 0.3 and len(variant) < 200:
            variant.append(_add_random_block(v, k))
        population.append(variant)
    
    # Evaluate population
    def eval_pop(pop):
        results = []
        for blocks in pop:
            unc = _fitness(blocks, v, k, t, subsets, subset_to_idx, n_subsets)
            results.append((unc, blocks))
        return results
    
    pop_results = eval_pop(population)
    pop_results.sort(key=lambda x: x[0])
    
    # GA loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Selection: keep top half
        top_half = pop_results[:len(pop_results)//2]
        
        # Create new population
        new_pop = []
        
        # Elitism: keep the best
        new_pop.append(top_half[0][1][:])
        
        # Crossover and mutation
        while len(new_pop) < pop_size:
            if len(top_half) < 2:
                # Clone with mutation
                parent = top_half[0][1][:]
                for i in range(len(parent)):
                    if random.random() < 0.1:
                        parent[i] = _mutate(parent[i], v, k, t, mutation_rate=0.5)
                new_pop.append(parent)
            else:
                # Pick two parents
                p1_idx = random.randint(0, len(top_half) - 1)
                p2_idx = random.randint(0, len(top_half) - 1)
                p1 = top_half[p1_idx][1]
                p2 = top_half[p2_idx][1]
                
                # Crossover
                child = _crossover(p1, p2)
                
                # Mutate
                if child:
                    for i in range(len(child)):
                        if random.random() < 0.15:
                            child[i] = _mutate(child[i], v, k, t, mutation_rate=0.4)
                    
                    # Occasionally add or remove a block
                    if random.random() < 0.2 and len(child) < 200:
                        child.append(_add_random_block(v, k))
                    elif random.random() < 0.15 and len(child) > 50:
                        child = _remove_random_block(child)
                
                new_pop.append(child)
        
        # Evaluate new population
        pop_results = eval_pop(new_pop)
        pop_results.sort(key=lambda x: x[0])
        
        # Update best
        if pop_results[0][0] < best_uncovered:
            best_uncovered = pop_results[0][0]
            best_blocks = pop_results[0][1][:]
        
        population = new_pop
    
    # Final local improvement: try to remove redundant blocks
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_blocks)):
            if time.time() > deadline:
                break
            # Try removing block i
            candidate = best_blocks[:i] + best_blocks[i+1:]
            unc = _fitness(candidate, v, k, t, subsets, subset_to_idx, n_subsets)
            if unc <= best_uncovered:
                if unc < best_uncovered:
                    best_uncovered = unc
                best_blocks = candidate
                improved = True
                break
    
    return best_blocks
