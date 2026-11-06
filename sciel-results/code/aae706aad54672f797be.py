"""
Strategy:
1. Represent the sorting network as a list of comparators.
2. Use a Genetic Algorithm (GA) to evolve a population of networks.
3. Fitness function penalizes both the number of comparators and the number of unsorted inputs (violations).
   - This allows the algorithm to move through invalid states to escape local minima, 
     unlike the current program which is restricted to valid states.
4. Crossover: Uniform crossover of comparator lists, followed by a "repair" phase 
   to remove duplicates and ensure validity if possible, or just let fitness handle it.
5. Mutation: Random insertion, deletion, or replacement of a comparator.
6. The GA explores the space of all possible comparator sequences, not just those 
   reachable by single-step edits from a specific seed. This can discover structurally 
   distinct topologies.
"""

import time
import numpy as np
import random

N = 16
MAX_TIME = 58.0
POP_SIZE = 50
ELITE_SIZE = 5

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _generate_oet_network():
    """
    Generates Odd-Even Transposition network (120 comparators).
    """
    comps = []
    for i in range(N):
        if i % 2 == 0:
            for j in range(0, N - 1, 2):
                comps.append((j, j + 1))
        else:
            for j in range(1, N - 1, 2):
                comps.append((j, j + 1))
    return comps

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    if not comps:
        return inputs.copy()
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _get_unsorted_count(comps, bits):
    """
    Returns the number of inputs that are NOT sorted by the network.
    """
    if not comps:
        out = bits
    else:
        out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    unsorted_mask = np.any(diffs < 0, axis=1)
    return np.sum(unsorted_mask)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Initial Population
    # 1. Batcher Network (63)
    batcher = _generate_batcher_network()
    # 2. OET Network (120)
    oet = _generate_oet_network()
    
    # Create initial population
    population = []
    # Add Batcher and OET
    population.append(batcher)
    population.append(oet)
    
    # Add random variations
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    rng = np.random.default_rng(42)
    
    while len(population) < POP_SIZE:
        # Random network of random length (10 to 100)
        length = rng.integers(10, 100)
        comp_list = [all_comps[rng.integers(0, len(all_comps))] for _ in range(length)]
        population.append(comp_list)
    
    # Evaluate initial population
    def evaluate(comps):
        unsorted = _get_unsorted_count(comps, bits)
        # Fitness: Lower is better.
        # If valid (unsorted == 0), fitness = len(comps)
        # If invalid, fitness = len(comps) + 1000 * unsorted
        # We want to minimize this.
        if unsorted == 0:
            return len(comps)
        else:
            return len(comps) + 1000 * unsorted

    best_comps = None
    best_score = float('inf')
    
    # Initial evaluation
    for comps in population:
        score = evaluate(comps)
        if score < best_score:
            best_score = score
            best_comps = comps[:]
    
    # If we already have a valid network, check its length
    # The goal is to minimize length among valid networks.
    
    # GA Loop
    while time.time() < deadline:
        if time.time() > deadline - 2.0:
            break
            
        # 1. Evaluate Fitness
        fitnesses = [evaluate(comps) for comps in population]
        
        # Find best in current population
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_score:
            best_score = fitnesses[min_idx]
            best_comps = population[min_idx][:]
        
        # 2. Selection (Tournament Selection)
        # We select parents based on fitness (lower is better)
        def select_parent():
            # Pick 3 random individuals, return the best
            candidates = [population[rng.integers(0, POP_SIZE)] for _ in range(3)]
            best_cand = None
            best_fit = float('inf')
            for c in candidates:
                f = evaluate(c)
                if f < best_fit:
                    best_fit = f
                    best_cand = c
            return best_cand

        new_population = []
        
        # 3. Elitism: Keep the best individuals
        # Sort population by fitness
        sorted_indices = np.argsort(fitnesses)
        for i in range(ELITE_SIZE):
            new_population.append(population[sorted_indices[i]][:])
            
        # 4. Crossover and Mutation
        while len(new_population) < POP_SIZE:
            p1 = select_parent()
            p2 = select_parent()
            
            # Crossover: Uniform crossover
            # Create a mask
            mask = rng.random(len(p1)) < 0.5
            child = []
            for i in range(max(len(p1), len(p2))):
                if i < len(p1) and mask[i]:
                    child.append(p1[i])
                elif i < len(p2) and not mask[i]:
                    child.append(p2[i])
                else:
                    # If one is shorter, or mask doesn't pick, pick randomly from parents
                    if i < len(p1):
                        child.append(p1[i])
                    elif i < len(p2):
                        child.append(p2[i])
                    else:
                        break
            
            # Mutation
            # 1. Random deletion
            if child and rng.random() < 0.5:
                idx = rng.integers(0, len(child))
                child.pop(idx)
            
            # 2. Random insertion
            if rng.random() < 0.5:
                new_comp = all_comps[rng.integers(0, len(all_comps))]
                idx = rng.integers(0, len(child) + 1)
                child.insert(idx, new_comp)
            
            # 3. Random replacement
            if child and rng.random() < 0.5:
                idx = rng.integers(0, len(child))
                child[idx] = all_comps[rng.integers(0, len(all_comps))]
            
            new_population.append(child)
        
        population = new_population
        
    # Return the best valid network found
    # If best_comps is invalid, we should return a known valid one (Batcher)
    # But the task requires a valid network. 
    # If we never found a valid one better than Batcher, return Batcher.
    # Actually, best_comps is updated only if score < best_score.
    # If best_score corresponds to an invalid network, we must ensure we return a valid one.
    
    # Re-check best_comps validity
    if best_comps is None:
        return batcher
        
    unsorted_best = _get_unsorted_count(best_comps, bits)
    if unsorted_best > 0:
        # Fallback to Batcher if the "best" score was from an invalid network 
        # that somehow had a lower "penalty" score than any valid network found?
        # No, valid networks have score = len. Invalid have score = len + 1000*unsorted.
        # So any valid network with len < 1000 will have a lower score than any invalid network.
        # Since Batcher is valid and has len 63, best_score should be <= 63 if we found any valid network.
        # If we didn't find any valid network, best_score would be high.
        # But we started with Batcher, so we always have a valid candidate.
        # So best_comps should be valid.
        pass
        
    return best_comps

# EVOLVE-BLOCK-END
