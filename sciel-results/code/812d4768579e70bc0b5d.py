"""Genetic Algorithm for covering design C(24,7,3)."""
import itertools
import random
import time
import sys
import numpy as np

# EVOLVE-BLOCK-START

def _greedy_covering(v, k, t, rng, time_limit):
    """Fast greedy covering to seed the GA population."""
    deadline = time.time() + time_limit
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    while uncovered and time.time() < deadline:
        first_uncovered = min(uncovered)
        block = list(first_uncovered)
        
        while len(block) < k:
            candidates = [x for x in range(v) if x not in block]
            if not candidates:
                break
            
            best_x = None
            best_gain = -1
            
            for x in candidates:
                gain = 0
                for S in itertools.combinations(block, t - 1):
                    if x not in S:
                        tset = tuple(sorted(S + (x,)))
                        if tset in uncovered:
                            gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None:
                break
            block.append(best_x)
        
        if len(block) < k:
            break
        
        block.sort()
        blocks.append(tuple(block))
        covered = set(itertools.combinations(block, t))
        uncovered -= covered
    
    return [list(b) for b in blocks]


def _coverage_vector(blocks, v, t, all_tsets):
    """Return a numpy array of coverage counts for each t-set."""
    cov = np.zeros(len(all_tsets), dtype=np.int32)
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    for b in blocks:
        for ts in itertools.combinations(b, t):
            idx = tset_to_idx.get(ts)
            if idx is not None:
                cov[idx] += 1
    return cov


def _fitness(cov):
    """
    Fitness: sum of inverse coverage counts for uncovered triples.
    Higher is better (lower uncovered penalty).
    Also penalize number of blocks indirectly via redundancy.
    """
    uncovered = cov[cov == 0]
    if len(uncovered) > 0:
        # Penalty for uncovered triples - very high
        return -1e6 - len(uncovered)
    
    # All covered: reward based on how 'tight' the covering is
    # Fewer redundant coverings is better
    total_cov = cov.sum()
    # Ideal: each triple covered exactly once
    # Penalty for over-coverage
    excess = total_cov - len(cov)
    return -excess  # Lower excess is better (more negative = worse)


def _crossover(parent1, parent2, v, k, t, all_tsets, tset_to_idx, rng):
    """
    Crossover: take a random subset of blocks from each parent,
    then fill gaps with greedy completion.
    """
    n1 = len(parent1)
    n2 = len(parent2)
    
    # Randomly select blocks from each parent
    # Target: roughly the same total number
    target_size = max(n1, n2)
    
    # Select from parent1
    if n1 > 0:
        frac1 = rng.uniform(0.3, 0.7)
        n_from_1 = max(1, min(n1, int(target_size * frac1)))
        idx1 = rng.sample(range(n1), n_from_1)
        child_blocks = [list(parent1[i]) for i in idx1]
    else:
        child_blocks = []
    
    # Select from parent2
    if n2 > 0:
        remaining = target_size - len(child_blocks)
        if remaining > 0:
            n_from_2 = min(n2, remaining)
            idx2 = rng.sample(range(n2), n_from_2)
            for i in idx2:
                child_blocks.append(list(parent2[i]))
    
    # Remove duplicates
    seen = set()
    unique_blocks = []
    for b in child_blocks:
        bt = tuple(sorted(b))
        if bt not in seen:
            seen.add(bt)
            unique_blocks.append(b)
    child_blocks = unique_blocks
    
    # Greedy fill: cover remaining uncovered triples
    if time.time() < _deadline:
        cov = _coverage_vector(child_blocks, v, t, all_tsets)
        uncovered_indices = np.where(cov == 0)[0]
        
        if len(uncovered_indices) > 0:
            # Greedily add blocks to cover uncovered triples
            max_add = 20  # Limit additions to keep population manageable
            for _ in range(max_add):
                if len(uncovered_indices) == 0:
                    break
                
                # Pick a random uncovered triple
                triple_idx = rng.choice(uncovered_indices)
                triple = all_tsets[triple_idx]
                
                # Try to find a good block containing this triple
                best_block = None
                best_gain = -1
                
                # Generate candidates by extending the triple
                base = list(triple)
                candidates = [x for x in range(v) if x not in base]
                
                # Try a few random extensions
                for _ in range(10):
                    if len(base) >= k:
                        break
                    x = rng.choice(candidates)
                    candidates.remove(x)
                    base.append(x)
                
                if len(base) < k:
                    # Fill remaining randomly
                    remaining_cands = [x for x in range(v) if x not in base]
                    while len(base) < k and remaining_cands:
                        x = rng.choice(remaining_cands)
                        remaining_cands.remove(x)
                        base.append(x)
                
                base.sort()
                
                # Calculate gain
                gain = 0
                for ts in itertools.combinations(base, t):
                    idx = tset_to_idx.get(ts)
                    if idx is not None and cov[idx] == 0:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = base
                
                if best_block is not None and best_gain > 0:
                    child_blocks.append(best_block)
                    # Update coverage
                    for ts in itertools.combinations(best_block, t):
                        idx = tset_to_idx.get(ts)
                        if idx is not None:
                            cov[idx] += 1
                    
                    # Recalculate uncovered
                    uncovered_indices = np.where(cov == 0)[0]
    
    return child_blocks


def _mutate(blocks, v, k, t, all_tsets, tset_to_idx, rng, mutation_rate=0.1):
    """
    Mutation: randomly replace or add blocks.
    """
    n = len(blocks)
    if n == 0:
        return blocks
    
    mutated = [list(b) for b in blocks]
    
    # Randomly replace some blocks
    num_replacements = max(1, int(n * mutation_rate))
    for _ in range(num_replacements):
        if not mutated:
            break
        idx = rng.randrange(len(mutated))
        
        # Generate a new random block
        new_block = sorted(rng.sample(range(v), k))
        mutated[idx] = new_block
    
    # Occasionally add a block
    if rng.random() < 0.2:
        new_block = sorted(rng.sample(range(v), k))
        mutated.append(new_block)
    
    # Occasionally remove a block (if covering remains valid)
    if rng.random() < 0.1 and len(mutated) > 1:
        # Check if we can safely remove
        idx = rng.randrange(len(mutated))
        cov = _coverage_vector(mutated, v, t, all_tsets)
        # Temporarily remove
        temp_blocks = mutated[:idx] + mutated[idx+1:]
        cov2 = _coverage_vector(temp_blocks, v, t, all_tsets)
        if not (cov2 == 0).any():
            mutated = temp_blocks
    
    return mutated


def _local_search(blocks, v, k, t, all_tsets, tset_to_idx, rng, time_limit):
    """Local search: try to remove redundant blocks and swap for better coverage."""
    deadline = time.time() + time_limit
    blocks = [list(b) for b in blocks]
    
    if not blocks:
        return blocks
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(blocks)
        if n == 0:
            break
        
        # Try to remove redundant blocks
        for idx in range(n):
            if time.time() >= deadline:
                break
            
            # Check if removing this block still covers everything
            temp_blocks = blocks[:idx] + blocks[idx+1:]
            cov = _coverage_vector(temp_blocks, v, t, all_tsets)
            if not (cov == 0).any():
                # Safe to remove
                blocks.pop(idx)
                improved = True
                break
        
        if improved:
            continue
        
        # Try swaps: replace one block with another
        for idx in range(len(blocks)):
            if time.time() >= deadline:
                break
            
            old_block = blocks[idx]
            
            # Try a few random replacements
            for _ in range(20):
                if time.time() >= deadline:
                    break
                
                new_block = sorted(rng.sample(range(v), k))
                
                # Check if replacement maintains coverage
                temp_blocks = blocks[:idx] + [new_block] + blocks[idx+1:]
                cov = _coverage_vector(temp_blocks, v, t, all_tsets)
                if not (cov == 0).any():
                    # Check if it reduces total coverage (tighter)
                    old_cov = _coverage_vector(blocks, v, t, all_tsets)
                    if cov.sum() < old_cov.sum():
                        blocks[idx] = new_block
                        improved = True
                        break
    
    return blocks


def _verify_covering(blocks, v, t, all_tsets):
    """Verify that every t-subset is covered by at least one block."""
    cov = _coverage_vector(blocks, v, t, all_tsets)
    return not (cov == 0).any()


def construct(v, k, t):
    global _deadline
    start_time = time.time()
    total_budget = 140.0
    _deadline = start_time + total_budget - 5.0  # Safety margin
    
    # Precompute all t-sets
    all_tsets = list(itertools.combinations(range(v), t))
    tset_to_idx = {ts: i for i, ts in enumerate(all_tsets)}
    
    # Phase 1: Generate initial population using greedy with different seeds
    pop_size = 8
    population = []
    
    greedy_time_each = (total_budget * 0.2) / pop_size
    for i in range(pop_size):
        rng = random.Random(1000 + i * 137)
        blocks = _greedy_covering(v, k, t, rng, greedy_time_each)
        population.append(blocks)
    
    # Evaluate initial fitness
    def get_fitness(blocks):
        cov = _coverage_vector(blocks, v, t, all_tsets)
        return _fitness(cov)
    
    # Phase 2: Genetic Algorithm
    ga_time_budget = total_budget * 0.6
    ga_deadline = start_time + (total_budget * 0.8)
    
    best_blocks = None
    best_fitness = float('-inf')
    
    # Track best across generations
    for gen in range(50):  # Max generations, but time-limited
        if time.time() >= ga_deadline:
            break
        
        # Evaluate fitness
        fitnesses = []
        for blocks in population:
            f = get_fitness(blocks)
            fitnesses.append(f)
            
            # Update best
            if f > best_fitness:
                best_fitness = f
                best_blocks = [list(b) for b in blocks]
        
        # Sort by fitness (descending)
        sorted_idx = np.argsort(fitnesses)[::-1]
        population = [population[i] for i in sorted_idx]
        
        # Elitism: keep top 2
        elite = [list(b) for b in population[:2]]
        
        # Generate next generation
        next_population = []
        
        # Keep elites
        next_population.extend(elite)
        
        # Generate offspring
        while len(next_population) < pop_size and time.time() < ga_deadline:
            # Select parents (tournament selection)
            p1_idx = rng_tournament(fitnesses, population, 3)
            p2_idx = rng_tournament(fitnesses, population, 3)
            
            # Crossover
            child = _crossover(population[p1_idx], population[p2_idx], 
                              v, k, t, all_tsets, tset_to_idx, 
                              random.Random(gen * 100 + len(next_population)))
            
            # Mutation
            child = _mutate(child, v, k, t, all_tsets, tset_to_idx,
                           random.Random(gen * 200 + len(next_population)),
                           mutation_rate=0.05)
            
            next_population.append(child)
        
        population = next_population
    
    # Phase 3: Local search on best solution
    remaining = total_budget - (time.time() - start_time) - 5.0
    if remaining > 10.0 and best_blocks is not None:
        rng = random.Random(99999)
        refined = _local_search(best_blocks, v, k, t, all_tsets, tset_to_idx, rng, remaining)
        
        if _verify_covering(refined, v, t, all_tsets):
            if len(refined) < len(best_blocks):
                best_blocks = refined
    
    # Final safety check
    if time.time() - start_time > total_budget - 5.0:
        return best_blocks
    
    # Ensure validity
    if best_blocks is not None and not _verify_covering(best_blocks, v, t, all_tsets):
        # Fallback: run greedy again
        rng = random.Random(1)
        best_blocks = _greedy_covering(v, k, t, rng, 10.0)
    
    return best_blocks


def rng_tournament(fitnesses, population, k):
    """Tournament selection: pick k random individuals, return the best."""
    n = len(population)
    candidates = random.sample(range(n), min(k, n))
    best_idx = candidates[0]
    for idx in candidates[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:
            best_idx = idx
    return best_idx

# EVOLVE-BLOCK-END
