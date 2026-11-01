"""Covering design C(24,7,3) via greedy with efficient gain computation, local search, and genetic evolution."""
import itertools
import time
import random
from collections import defaultdict
import numpy as np

def _prune_redundant_blocks(blocks, v, t):
    """
    Removes blocks that do not cover any unique t-subset.
    Returns the pruned list of blocks.
    """
    if not blocks:
        return blocks
    
    # Precompute t-subsets for each block
    block_tsets = []
    for b in blocks:
        tsets = [frozenset(c) for c in itertools.combinations(b, t)]
        block_tsets.append(tsets)
        
    # Count how many blocks cover each t-set
    tset_to_blocks = defaultdict(list)
    for i, tsets in enumerate(block_tsets):
        for ts in tsets:
            tset_to_blocks[ts].append(i)
            
    to_remove = set()
    for i, tsets in enumerate(block_tsets):
        is_redundant = True
        for ts in tsets:
            if len(tset_to_blocks[ts]) == 1:
                is_redundant = False
                break
        if is_redundant:
            to_remove.add(i)
            
    if to_remove:
        new_blocks = [b for i, b in enumerate(blocks) if i not in to_remove]
        return new_blocks
    else:
        return blocks


def _compute_fitness(blocks, v, t, all_tsets):
    """
    Compute a smooth fitness score for a block set.
    Lower is better. Uses sum of inverse coverage counts for uncovered triples
    plus a penalty for number of blocks.
    """
    if not blocks:
        return float('inf')
    
    # Count coverage for each t-set
    coverage = np.zeros(len(all_tsets), dtype=np.int32)
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    for b in blocks:
        for c in itertools.combinations(b, t):
            ts = frozenset(c)
            if ts in tset_index:
                coverage[tset_index[ts]] += 1
    
    uncovered_count = np.sum(coverage == 0)
    if uncovered_count > 0:
        # Penalty for uncovered triples: sum of 1/coverage for uncovered (which is inf)
        # Instead, use a large penalty per uncovered triple
        return 10000.0 * uncovered_count + len(blocks)
    else:
        # All covered: minimize number of blocks, with small bonus for redundancy
        return float(len(blocks))


def _ga_crossover(parent1, parent2, v, k, t, all_tsets, deadline):
    """
    Crossover: combine blocks from two parents, then prune and repair.
    """
    if time.time() > deadline - 5:
        return parent1
    
    # Start with union of blocks from both parents
    combined = list(parent1)
    for b in parent2:
        if b not in combined:
            combined.append(b)
    
    # Prune redundant blocks
    pruned = _prune_redundant_blocks(combined, v, t)
    
    # Check if still valid
    uncovered = set(all_tsets)
    for b in pruned:
        for c in itertools.combinations(b, t):
            uncovered.discard(frozenset(c))
    
    if uncovered:
        # Repair: add blocks for uncovered triples
        for ts in list(uncovered):
            if time.time() > deadline - 3:
                break
            block = list(ts)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block:
                    block.append(x)
            block.sort()
            if block not in pruned:
                pruned.append(block)
                for c in itertools.combinations(block, t):
                    uncovered.discard(frozenset(c))
    
    if not uncovered:
        return pruned
    return parent1


def _ga_mutation(blocks, v, k, t, all_tsets, deadline):
    """
    Mutation: randomly replace some blocks with new ones, then prune and repair.
    """
    if time.time() > deadline - 5:
        return blocks
    
    mutated = list(blocks)
    n_mutate = max(1, len(mutated) // 10)
    
    for _ in range(n_mutate):
        if time.time() > deadline - 3:
            break
        # Remove a random block
        idx = random.randint(0, len(mutated) - 1)
        mutated.pop(idx)
        
        # Add a new random block
        new_block = sorted(random.sample(range(v), k))
        if new_block not in mutated:
            mutated.append(new_block)
    
    # Prune
    pruned = _prune_redundant_blocks(mutated, v, t)
    
    # Check validity and repair
    uncovered = set(all_tsets)
    for b in pruned:
        for c in itertools.combinations(b, t):
            uncovered.discard(frozenset(c))
    
    if uncovered:
        for ts in list(uncovered):
            if time.time() > deadline - 2:
                break
            block = list(ts)
            while len(block) < k:
                x = random.randint(0, v - 1)
                if x not in block:
                    block.append(x)
            block.sort()
            if block not in pruned:
                pruned.append(block)
                for c in itertools.combinations(block, t):
                    uncovered.discard(frozenset(c))
    
    if not uncovered:
        return pruned
    return blocks

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # Precompute all t-subsets as frozensets for fast lookup
    all_tsets = [frozenset(c) for c in itertools.combinations(range(v), t)]
    tset_index = {ts: i for i, ts in enumerate(all_tsets)}
    
    best_blocks = None
    best_count = float('inf')
    
    # Phase 1: Multi-start greedy to build initial population
    population = []
    
    while time.time() < deadline - 60:
        # Greedy Construction
        uncovered = set(all_tsets)
        blocks = []
        
        while uncovered and time.time() < deadline - 5:
            seed_tset = random.choice(list(uncovered))
            block = list(seed_tset)
            
            while len(block) < k and time.time() < deadline - 5:
                block_set = set(block)
                best_gain = -1
                best_candidates = []
                
                for x in range(v):
                    if x in block_set:
                        continue
                    gain = 0
                    for R in itertools.combinations(block, t - 1):
                        ts = frozenset(list(R) + [x])
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
            
            if len(block) == k:
                block.sort()
                covered = set()
                for c in itertools.combinations(block, t):
                    covered.add(frozenset(c))
                uncovered -= covered
                blocks.append(block)
        
        # Check if we have full coverage
        if uncovered:
            for ts in list(uncovered):
                if time.time() >= deadline - 2:
                    break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v - 1)
                    if x not in block:
                        block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
        
        if uncovered:
            continue
            
        # Prune
        pruned_blocks = _prune_redundant_blocks(blocks, v, t)
        prev_len = len(pruned_blocks)
        while len(pruned_blocks) < prev_len and time.time() < deadline - 1:
            prev_len = len(pruned_blocks)
            pruned_blocks = _prune_redundant_blocks(pruned_blocks, v, t)
            
        if len(pruned_blocks) < best_count:
            best_count = len(pruned_blocks)
            best_blocks = pruned_blocks
        
        population.append(pruned_blocks)
        
        # Keep population size manageable
        if len(population) >= 10:
            population.pop(0)
        
        if time.time() > deadline - 65:
            break

    # Phase 2: Genetic Algorithm evolution
    if population and time.time() < deadline - 10:
        # Sort population by size
        population.sort(key=lambda b: len(b))
        
        ga_deadline = deadline - 5
        generation = 0
        
        while time.time() < ga_deadline:
            generation += 1
            new_population = []
            
            # Elitism: keep top half
            elite_size = max(1, len(population) // 2)
            for i in range(min(elite_size, len(population))):
                new_population.append(population[i])
            
            # Generate offspring via crossover and mutation
            while len(new_population) < len(population) and time.time() < ga_deadline:
                # Select parents (tournament selection)
                p1_idx = random.randint(0, len(population) - 1)
                p2_idx = random.randint(0, len(population) - 1)
                
                if random.random() < 0.6:
                    # Crossover
                    child = _ga_crossover(population[p1_idx], population[p2_idx], v, k, t, all_tsets, ga_deadline)
                else:
                    # Mutation
                    child = _ga_mutation(population[p1_idx], v, k, t, all_tsets, ga_deadline)
                
                if child:
                    # Prune the child
                    child = _prune_redundant_blocks(child, v, t)
                    prev_len = len(child)
                    while len(child) < prev_len and time.time() < ga_deadline - 1:
                        prev_len = len(child)
                        child = _prune_redundant_blocks(child, v, t)
                    
                    # Check validity
                    uncovered = set(all_tsets)
                    for b in child:
                        for c in itertools.combinations(b, t):
                            uncovered.discard(frozenset(c))
                    
                    if not uncovered:
                        new_population.append(child)
                        if len(child) < best_count:
                            best_count = len(child)
                            best_blocks = child
            
            if not new_population:
                break
            
            population = new_population
            population.sort(key=lambda b: len(b))
            
            # Stop if we've hit the known best
            if best_count <= 76:
                break

    # Final fallback if no valid solution found
    if best_blocks is None:
        uncovered = set(all_tsets)
        blocks = []
        while uncovered and time.time() < deadline - 1:
            seed_tset = next(iter(uncovered))
            block = list(seed_tset)
            while len(block) < k:
                block_set = set(block)
                best_x = -1
                best_gain = -1
                for x in range(v):
                    if x in block_set: continue
                    gain = sum(1 for R in itertools.combinations(block, t-1) if frozenset(list(R)+[x]) in uncovered)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x == -1:
                    rem = [x for x in range(v) if x not in block_set]
                    if rem: best_x = rem[0]
                    else: break
                block.append(best_x)
            if len(block) == k:
                block.sort()
                covered = set(frozenset(c) for c in itertools.combinations(block, t))
                uncovered -= covered
                blocks.append(block)
        
        if not uncovered:
            best_blocks = blocks
        else:
            for ts in list(uncovered):
                if time.time() >= deadline - 0.5: break
                block = list(ts)
                while len(block) < k:
                    x = random.randint(0, v-1)
                    if x not in block: block.append(x)
                block.sort()
                if block not in blocks:
                    blocks.append(block)
                    for c in itertools.combinations(block, t):
                        uncovered.discard(frozenset(c))
            best_blocks = blocks

    return best_blocks
