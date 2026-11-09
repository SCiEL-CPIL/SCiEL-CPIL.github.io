"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

import time
import random
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _apply_network(network, inputs):
    """Apply compare-exchange network to an array of values.
    network: list of (i,j) pairs
    inputs: array-like of N values
    Returns the sorted output as a list.
    """
    vals = list(inputs)
    for i, j in network:
        if vals[i] > vals[j]:
            vals[i], vals[j] = vals[j], vals[i]
    return vals

def _is_sorting_network(network):
    """Check if network sorts all 2^16 zero-one inputs using zero-one principle."""
    # For efficiency, we can use a smarter check, but for 16 wires we can do full check
    # with early termination
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        outputs = _apply_network(network, inputs)
        # Check if outputs are sorted (non-decreasing)
        for k in range(N - 1):
            if outputs[k] > outputs[k + 1]:
                return False
    return True

def _count_inversions(inputs):
    """Count inversions in a list of 0/1 values."""
    inv = 0
    for i in range(len(inputs)):
        for j in range(i + 1, len(inputs)):
            if inputs[i] > inputs[j]:
                inv += 1
    return inv

def _fitness(network):
    """
    Compute fitness: number of zero-one inputs that are NOT correctly sorted.
    Lower is better. 0 means perfect sorting network.
    Also penalize by total comparator count to encourage smaller networks.
    """
    # Sample-based fitness for speed: check a random subset of inputs
    # But for correctness we need to be thorough. Let's use all 65536 but vectorize.
    
    # Use numpy for speed: represent network as array of (i,j) pairs
    # Process all 65536 inputs at once in batches
    all_masks = np.arange(1 << N, dtype=np.int32)
    # Extract bits: inputs[m, i] = (m >> i) & 1
    # Shape: (65536, 16)
    inputs_arr = ((all_masks[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int32)
    
    # Apply network
    for i, j in network:
        # Swap where inputs_arr[:, i] > inputs_arr[:, j]
        mask_swap = inputs_arr[:, i] > inputs_arr[:, j]
        # Swap values
        tmp = inputs_arr[mask_swap, i].copy()
        inputs_arr[mask_swap, i] = inputs_arr[mask_swap, j]
        inputs_arr[mask_swap, j] = tmp
    
    # Check sortedness: for each row, all elements should be non-decreasing
    # i.e., inputs_arr[:, k] <= inputs_arr[:, k+1] for all k
    diffs = np.diff(inputs_arr, axis=1)  # shape (65536, 15)
    # Unsorted if any diff < 0
    unsorted_rows = np.any(diffs < 0, axis=1)
    return int(np.sum(unsorted_rows))

def _construct_impl():
    start_time = time.time()
    DEADLINE = 75  # seconds, leave 15s margin for safety
    
    # Seed: odd-even transposition (120 comparators)
    seed_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    seed_fitness = _fitness(seed_net)
    assert seed_fitness == 0, "Seed network should be valid"
    
    best_net = seed_net
    best_fitness = 0  # Since seed is valid
    best_count = len(seed_net)
    
    random.seed(42)
    
    # Genetic algorithm:
    # Represent network as list of comparator pairs.
    # Operations: 
    #   - Remove a comparator (if network remains valid, keep)
    #   - Swap order of two comparators
    #   - Replace a comparator with a different pair
    #   - Insert a comparator
    
    pop_size = 20
    population = [(0, seed_net)]  # (fitness, network)
    
    # Add some random mutations to seed
    for _ in range(pop_size - 1):
        net = seed_net[:]
        # Random mutations
        for _ in range(3):
            op = random.choice(['remove', 'swap', 'replace', 'insert'])
            if op == 'remove' and len(net) > 1:
                idx = random.randint(0, len(net) - 1)
                net.pop(idx)
            elif op == 'swap' and len(net) > 1:
                i, j = random.sample(range(len(net)), 2)
                net[i], net[j] = net[j], net[i]
            elif op == 'replace' and len(net) > 0:
                idx = random.randint(0, len(net) - 1)
                a, b = random.sample(range(N), 2)
                net[idx] = (min(a, b), max(a, b))
            elif op == 'insert':
                a, b = random.sample(range(N), 2)
                idx = random.randint(0, len(net))
                net.insert(idx, (min(a, b), max(a, b)))
        fitness = _fitness(net)
        population.append((fitness, net))
    
    # Sort population by fitness
    population.sort(key=lambda x: x[0])
    
    generations = 0
    while time.time() - start_time < DEADLINE and generations < 500:
        generations += 1
        population.sort(key=lambda x: x[0])
        
        # Elitism: keep top half
        parents = population[:pop_size // 2]
        
        new_pop = []
        # Keep the best
        new_pop.append(parents[0])
        
        while len(new_pop) < pop_size and time.time() - start_time < DEADLINE:
            p1_idx = random.randint(0, len(parents) - 1)
            p2_idx = random.randint(0, len(parents) - 1)
            p1 = parents[p1_idx][1]
            p2 = parents[p2_idx][1]
            
            # Crossover: take prefix from p1 and suffix from p2
            if len(p1) < 2 or len(p2) < 2:
                child = p1[:]
            else:
                cut1 = random.randint(1, len(p1) - 1)
                cut2 = random.randint(1, len(p2) - 1)
                child = p1[:cut1] + p2[cut2:]
            
            # Mutation
            for _ in range(random.randint(1, 4)):
                op = random.choice(['remove', 'swap', 'replace', 'insert'])
                if op == 'remove' and len(child) > 1:
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
                elif op == 'swap' and len(child) > 1:
                    i, j = random.sample(range(len(child)), 2)
                    child[i], child[j] = child[j], child[i]
                elif op == 'replace' and len(child) > 0:
                    idx = random.randint(0, len(child) - 1)
                    a, b = random.sample(range(N), 2)
                    child[idx] = (min(a, b), max(a, b))
                elif op == 'insert':
                    a, b = random.sample(range(N), 2)
                    idx = random.randint(0, len(child))
                    child.insert(idx, (min(a, b), max(a, b)))
            
            fitness = _fitness(child)
            if fitness == 0 and len(child) < best_count:
                best_count = len(child)
                best_net = child
            new_pop.append((fitness, child))
        
        population = new_pop
    
    return best_net

def construct():
    try:
        return _construct_impl()
    except Exception:
        # Fallback to seed
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
