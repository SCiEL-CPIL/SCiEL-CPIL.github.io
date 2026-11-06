"""
Strategy:
A constructive genetic algorithm that evolves network topologies from scratch,
rather than pruning a fixed baseline. The key insight is that the fitness landscape
for sorting networks has many local minima when viewed through the lens of "removal"
from a large valid network. A GA explores a broader space by combining partial
structures (crossover) and introducing new comparators (mutation), guided by a
fitness function that measures how close the network is to sorting all inputs.

The fitness function is based on the number of unsorted 0-1 patterns. For a network
with C comparators, we can quickly estimate its "sorting power" by checking a subset
of inputs or by using a heuristic based on the number of inversions remaining.
However, to be rigorous, we must eventually check all 65,536 inputs.

To make this feasible within the time limit:
1. We use a fast, vectorized check for validity.
2. We maintain a population of candidate networks.
3. We use a "survival of the fittest" approach where fitness is primarily the
   negative comparator count (we want to minimize), but we only consider valid
   networks for the "best" tracking. For the GA, we can use a softer fitness
   that penalizes invalid networks heavily but allows them to survive if they
   have fewer comparators, hoping that crossover/mutation will lead to a valid
   network with even fewer comparators.

Actually, a simpler and more robust approach for the GA in this context:
- Initialize the population with a mix of known good networks (Batcher, etc.) and
  random networks.
- Fitness: If valid, fitness = -len(network). If invalid, fitness = -len(network) - penalty.
  The penalty should be large enough to discourage invalid networks but not so large
  that we never explore smaller invalid ones.
- Selection: Tournament selection based on fitness.
- Crossover: Uniform crossover of comparator lists.
- Mutation: Randomly add, remove, or swap comparators.
- Elitism: Keep the best valid network found so far.

The key difference from the current program is that we are not just removing
comparators from a valid network; we are actively constructing new networks by
combining parts of different networks and introducing random changes. This allows
us to explore topologies that might not be reachable by simple deletion.

To ensure we find a valid network, we will also include a local search phase
within the GA loop: if a network is valid, we try to remove comparators to make
it even smaller.

We will use a deadline guard to ensure we return within the time limit.
"""

import time
import numpy as np
import random

N = 16
MAX_TIME = 58.0
POP_SIZE = 50
TOURNAMENT_SIZE = 5

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

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _is_sorted_network_vectorized(comps, bits):
    """
    Check if the network sorts all 2^16 boolean inputs.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    return np.all(diffs >= 0)

def _fitness(comps, bits, penalty=1000):
    """
    Calculate fitness. Lower is better.
    If valid, fitness = len(comps).
    If invalid, fitness = len(comps) + penalty.
    """
    if _is_sorted_network_vectorized(comps, bits):
        return len(comps)
    else:
        return len(comps) + penalty

def _tournament_select(population, fitnesses, rng):
    """
    Select an individual using tournament selection.
    """
    indices = rng.integers(0, len(population), size=TOURNAMENT_SIZE)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx][:]

def _crossover(parent1, parent2, rng):
    """
    Uniform crossover of two comparator lists.
    """
    child = []
    for i in range(max(len(parent1), len(parent2))):
        if i < len(parent1) and (i >= len(parent2) or rng.random() < 0.5):
            child.append(parent1[i])
        elif i < len(parent2):
            child.append(parent2[i])
    # Ensure no duplicates? Not strictly necessary, but might help.
    # For now, keep duplicates as they represent "double" comparators which are redundant.
    return child

def _mutate(comps, rng, all_comps):
    """
    Mutate a comparator list.
    """
    if len(comps) == 0:
        return comps
    
    mutation_type = rng.integers(0, 3)
    
    if mutation_type == 0:
        # Remove a random comparator
        if len(comps) > 1:
            idx = rng.integers(0, len(comps))
            comps.pop(idx)
    elif mutation_type == 1:
        # Add a random comparator
        new_comp = all_comps[rng.integers(0, len(all_comps))]
        comps.append(new_comp)
    else:
        # Swap a random comparator with another
        if len(comps) > 1:
            idx = rng.integers(0, len(comps))
            old_comp = comps[idx]
            new_comp = all_comps[rng.integers(0, len(all_comps))]
            if new_comp != old_comp:
                comps[idx] = new_comp
    
    return comps

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # Pre-compute all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Initialize population
    batcher = _generate_batcher_network()
    population = []
    fitnesses = []
    
    # Add Batcher network and some variations
    for _ in range(POP_SIZE // 2):
        pop = batcher[:]
        pop = _mutate(pop, np.random.default_rng(), all_comps)
        population.append(pop)
    
    # Add random networks
    for _ in range(POP_SIZE // 2):
        size = random.randint(30, 80)
        pop = [all_comps[random.randint(0, len(all_comps)-1)] for _ in range(size)]
        population.append(pop)
    
    rng = np.random.default_rng(42)
    
    best_comps = batcher[:]
    best_fitness = _fitness(best_comps, bits)
    
    # Calculate initial fitnesses
    for i, pop in enumerate(population):
        fitnesses.append(_fitness(pop, bits))
    
    # Genetic Algorithm Loop
    generation = 0
    while time.time() < deadline:
        if time.time() > deadline - 2.0:
            break
        
        # Calculate fitnesses
        for i, pop in enumerate(population):
            fitnesses[i] = _fitness(pop, bits)
            
            # Update best
            if fitnesses[i] < best_fitness:
                best_fitness = fitnesses[i]
                best_comps = pop[:]
        
        # Create next generation
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = np.argmin(fitnesses)
        new_population.append(population[best_idx][:])
        
        # Fill the rest of the population
        while len(new_population) < POP_SIZE:
            if time.time() > deadline - 1.0:
                break
                
            parent1 = _tournament_select(population, fitnesses, rng)
            parent2 = _tournament_select(population, fitnesses, rng)
            
            child = _crossover(parent1, parent2, rng)
            child = _mutate(child, rng, all_comps)
            
            new_population.append(child)
        
        population = new_population
        generation += 1
        
        # Occasionally, try a local search on the best individual
        if generation % 5 == 0 and time.time() < deadline - 5.0:
            current_comps = best_comps[:]
            improved = True
            while improved and time.time() < deadline - 5.0:
                improved = False
                for _ in range(10): # Try 10 random removals
                    if len(current_comps) <= 1:
                        break
                    idx = rng.integers(0, len(current_comps))
                    removed = current_comps.pop(idx)
                    if _is_sorted_network_vectorized(current_comps, bits):
                        best_comps = current_comps
                        best_fitness = len(best_comps)
                        improved = True
                    else:
                        current_comps.insert(idx, removed)
    
    return best_comps

# EVOLVE-BLOCK-END
