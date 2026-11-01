"""Covering design C(24,7,3) via hybrid: fast greedy initialization + genetic algorithm refinement."""
import itertools
import time
import random
from collections import defaultdict
import numpy as np

def _fast_greedy_init(v, k, t, deadline, seed=None):
    """
    Fast greedy initialization using a randomized gain-based approach.
    Returns a list of blocks that covers all t-subsets (if possible within deadline).
    """
    if seed is not None:
        random.seed(seed)
    
    all_tsets = list(itertools.combinations(range(v), t))
    uncovered = set(all_tsets)
    blocks = []
    
    # Precompute for fast gain calculation
    # For each (t-1) subset, list of t-subsets containing it
    tm1_to_tsets = defaultdict(list)
    for ts in all_tsets:
        for i in range(t):
            tm1 = tuple(sorted(ts[:i] + ts[i+1:]))
            tm1_to_tsets[tm1].append(ts)
    
    while uncovered and time.time() < deadline:
        # Pick a random uncovered t-set to seed a new block
        seed_ts = random.choice(list(uncovered))
        block = list(seed_ts)
        block_set = set(block)
        
        while len(block) < k and time.time() < deadline:
            best_x = -1
            best_gain = -1
            best_candidates = []
            
            for x in range(v):
                if x in block_set:
                    continue
                gain = 0
                for R in itertools.combinations(block, t - 1):
                    ts = tuple(sorted(list(R) + [x]))
                    if ts in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_candidates = [x]
                elif gain == best_gain:
                    best_candidates.append(x)
            
            if not best_candidates:
                if best_gain == 0:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                else:
                    break
            else:
                best_x = random.choice(best_candidates)
            
            block.append(best_x)
            block_set.add(best_x)
        
        if len(block) == k:
            block.sort()
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
            if block not in blocks:
                blocks.append(block)
    
    # Ensure full coverage if possible
    if uncovered:
        for ts in list(uncovered):
            if time.time() >= deadline:
                break
            block = list(ts)
            block_set = set(block)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block_set:
                    block.append(x)
                    block_set.add(x)
            block.sort()
            if block not in blocks:
                blocks.append(block)
            covered = set(itertools.combinations(block, t))
            uncovered -= covered
    
    return blocks

def _prune_redundant(blocks, v, t):
    """Remove blocks that do not cover any unique t-subset."""
    if not blocks:
        return blocks
    
    tset_to_blocks = defaultdict(int)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            tset_to_blocks[ts] += 1
    
    to_remove = set()
    for i, b in enumerate(blocks):
        is_redundant = True
        for ts in itertools.combinations(b, t):
            if tset_to_blocks[ts] == 1:
                is_redundant = False
                break
        if is_redundant:
            to_remove.add(i)
    
    if to_remove:
        return [b for i, b in enumerate(blocks) if i not in to_remove]
    return blocks

def _compute_fitness(blocks, v, t):
    """
    Fitness: sum of inverse coverage counts for all t-subsets.
    Lower is better. Valid coverings have fitness = n_tsets.
    """
    coverage_count = defaultdict(int)
    for b in blocks:
        for ts in itertools.combinations(b, t):
            coverage_count[ts] += 1
    
    all_tsets = itertools.combinations(range(v), t)
    fitness = 0.0
    for ts in all_tsets:
        cnt = coverage_count.get(ts, 0)
        if cnt == 0:
            fitness += 1000.0  # Heavy penalty for uncovered
        else:
            fitness += 1.0 / cnt
    
    return fitness

def _mutate_block(block, v, k, mutation_rate=0.3):
    """
    Mutate a block by randomly swapping elements.
    """
    new_block = list(block)
    for _ in range(k):
        if random.random() < mutation_rate:
            old = new_block.pop(random.randint(0, k - 1))
            new = random.randint(0, v - 1)
            if new not in new_block:
                new_block.append(new)
            else:
                new_block.append(old)
    new_block = sorted(set(new_block))
    # Ensure size k
    while len(new_block) < k:
        x = random.randint(0, v - 1)
        if x not in new_block:
            new_block.append(x)
    new_block = sorted(new_block[:k])
    return new_block

def _crossover(block1, block2, v, k):
    """
    Crossover: combine elements from two blocks to form a new block.
    """
    set1 = set(block1)
    set2 = set(block2)
    common = set1 & set2
    only1 = set1 - set2
    only2 = set2 - set1
    
    # Start with common
    new_block = list(common)
    
    # Fill remaining slots with random mix
    while len(new_block) < k:
        if only1 and (not only2 or random.random() < 0.5):
            new_block.append(only1.pop())
        elif only2:
            new_block.append(only2.pop())
        else:
            x = random.randint(0, v - 1)
            if x not in new_block:
                new_block.append(x)
    
    return sorted(new_block[:k])

def _genetic_refine(blocks, v, k, t, deadline, pop_size=20, generations=10):
    """
    Genetic algorithm to refine a set of blocks.
    """
    if not blocks:
        return blocks
    
    current_best = list(blocks)
    best_fitness = _compute_fitness(current_best, v, t)
    
    population = [list(current_best)]
    # Add some variation
    for _ in range(pop_size - 1):
        mutated = [list(b) for b in current_best]
        # Randomly add/remove/modify blocks
        if random.random() < 0.5 and len(mutated) > 10:
            mutated.pop(random.randint(0, len(mutated) - 1))
        if random.random() < 0.5:
            # Add a new random block
            new_block = sorted(random.sample(range(v), k))
            mutated.append(new_block)
        for i in range(len(mutated)):
            if random.random() < 0.3:
                mutated[i] = _mutate_block(mutated[i], v, k)
        population.append(mutated)
    
    gen = 0
    while gen < generations and time.time() < deadline:
        gen += 1
        
        # Evaluate fitness
        fitnesses = [_compute_fitness(ind, v, t) for ind in population]
        
        # Find best
        best_idx = np.argmin(fitnesses)
        if fitnesses[best_idx] < best_fitness:
            best_fitness = fitnesses[best_idx]
            current_best = [list(b) for b in population[best_idx]]
        
        # Selection: keep top half
        sorted_indices = np.argsort(fitnesses)
        survivors = [population[i] for i in sorted_indices[:pop_size // 2]]
        
        # Generate next generation
        new_population = []
        while len(new_population) < pop_size:
            # Crossover
            p1 = random.choice(survivors)
            p2 = random.choice(survivors)
            
            # Create offspring by mixing blocks
            offspring = []
            for i in range(len(p1)):
                if random.random() < 0.5:
                    offspring.append(list(p1[i]))
                elif i < len(p2):
                    offspring.append(list(p2[i]))
                else:
                    offspring.append(list(p1[i]))
            
            # Mutate some blocks
            for i in range(len(offspring)):
                if random.random() < 0.2:
                    offspring[i] = _mutate_block(offspring[i], v, k)
            
            # Occasionally add/remove block
            if random.random() < 0.1 and len(offspring) > 10:
                offspring.pop(random.randint(0, len(offspring) - 1))
            if random.random() < 0.1:
                new_block = sorted(random.sample(range(v), k))
                offspring.append(new_block)
            
            new_population.append(offspring)
        
        population = new_population
    
    return current_best

def _local_search(blocks, v, k, t, deadline, max_iters=50):
    """
    Local search: try to replace blocks with alternatives that allow further pruning.
    """
    if not blocks:
        return blocks
    
    current_blocks = list(blocks)
    best_count = len(current_blocks)
    no_improve = 0
    
    while time.time() < deadline and no_improve < 10:
        improved = False
        n = len(current_blocks)
        
        # Compute coverage counts
        tset_to_blocks = defaultdict(int)
        for b in current_blocks:
            for ts in itertools.combinations(b, t):
                tset_to_blocks[ts] += 1
        
        # For each block, find its unique t-sets
        unique_tsets = []
        for b in current_blocks:
            uniq = [ts for ts in itertools.combinations(b, t) if tset_to_blocks[ts] == 1]
            unique_tsets.append(uniq)
        
        # Try to replace each block
        for i in range(n):
            if time.time() >= deadline:
                break
            
            if not unique_tsets[i]:
                continue  # Already redundant
            
            block_i = set(current_blocks[i])
            best_replacement = None
            best_score = -1
            
            # Try random modifications
            for _ in range(30):
                if time.time() >= deadline:
                    break
                
                cand = list(block_i)
                random.shuffle(cand)
                
                for _ in range(3):
                    if time.time() >= deadline:
                        break
                    if len(cand) < k:
                        break
                    pos = random.randint(0, len(cand) - 1)
                    removed = cand.pop(pos)
                    new_point = random.randint(0, v - 1)
                    if new_point not in cand:
                        cand.append(new_point)
                
                if len(cand) != k or len(set(cand)) != k:
                    continue
                
                cand.sort()
                if frozenset(cand) == frozenset(current_blocks[i]):
                    continue
                
                cand_set = set(cand)
                covers_all = True
                for ts in unique_tsets[i]:
                    if not set(ts).issubset(cand_set):
                        covers_all = False
                        break
                
                if not covers_all:
                    continue
                
                # Score: how many OTHER blocks' unique t-sets does cand also cover?
                score = 0
                cand_tsets = set(frozenset(c) for c in itertools.combinations(cand, t))
                for j in range(n):
                    if j == i:
                        continue
                    for ts in unique_tsets[j]:
                        if frozenset(ts) in cand_tsets:
                            score += 1
                
                if score > best_score:
                    best_score = score
                    best_replacement = cand
            
            if best_replacement is not None:
                test_blocks = list(current_blocks)
                test_blocks[i] = best_replacement
                
                pruned = _prune_redundant(test_blocks, v, t)
                prev_len = len(pruned)
                while len(pruned) < prev_len and time.time() < deadline:
                    prev_len = len(pruned)
                    pruned = _prune_redundant(pruned, v, t)
                
                if len(pruned) < best_count:
                    current_blocks = pruned
                    best_count = len(current_blocks)
                    improved = True
                    no_improve = 0
                    break
                else:
                    no_improve += 1
        
        if not improved:
            no_improve += 1
    
    return current_blocks

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start hybrid approach
    for restart in range(20):
        if time.time() > deadline - 30:
            break
        
        # 1. Fast greedy initialization
        init_blocks = _fast_greedy_init(v, k, t, deadline - 20, seed=restart * 1000)
        
        if not init_blocks:
            continue
        
        # Ensure validity
        all_tsets = set(itertools.combinations(range(v), t))
        covered = set()
        for b in init_blocks:
            covered.update(itertools.combinations(b, t))
        
        if not all_tsets.issubset(covered):
            # Force cover remaining
            uncovered = all_tsets - covered
            for ts in list(uncovered):
                if time.time() >= deadline - 10:
                    break
                block = list(ts)
                block_set = set(block)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block_set:
                        block.append(x)
                        block_set.add(x)
                block.sort()
                if block not in init_blocks:
                    init_blocks.append(block)
                covered.update(itertools.combinations(block, t))
        
        if not all_tsets.issubset(covered):
            continue
        
        # 2. Prune
        pruned = _prune_redundant(init_blocks, v, t)
        prev_len = len(pruned)
        while len(pruned) < prev_len and time.time() < deadline - 10:
            prev_len = len(pruned)
            pruned = _prune_redundant(pruned, v, t)
        
        # 3. Genetic refinement
        if time.time() < deadline - 15:
            ga_blocks = _genetic_refine(pruned, v, k, t, deadline - 10, pop_size=15, generations=5)
            if len(ga_blocks) < len(pruned):
                pruned = ga_blocks
        
        # 4. Local search
        if time.time() < deadline - 5:
            ls_blocks = _local_search(pruned, v, k, t, deadline - 2, max_iters=30)
            if len(ls_blocks) < len(pruned):
                pruned = ls_blocks
        
        if len(pruned) < best_count:
            best_count = len(pruned)
            best_blocks = list(pruned)
        
        if time.time() > deadline - 15:
            break
    
    if best_blocks is None or len(best_blocks) == 0:
        # Fallback: simple greedy
        all_tsets = set(itertools.combinations(range(v), t))
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 2:
            seed_ts = random.choice(list(uncovered))
            block = list(seed_ts)
            block_set = set(block)
            while len(block) < k:
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = tuple(sorted(list(R) + [x]))
                        if ts in uncovered:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    remaining = [x for x in range(v) if x not in block_set]
                    if remaining:
                        best_x = random.choice(remaining)
                    else:
                        break
                block.append(best_x)
                block_set.add(best_x)
            
            if len(block) == k:
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                covered = set(itertools.combinations(block, t))
                uncovered -= covered
        
        best_blocks = blocks
    
    return best_blocks
