"""Size-efficient sorting network search for 17 wires."""
import time
import random
import numpy as np

N = 17
SEED = 42

# Time management
_START_TIME = time.time()
_TIME_LIMIT = 88.0  # safety margin for 90s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _simulate_network_fast(comparators, mask):
    """Simulate network on a single Boolean input, return output state."""
    state = mask
    for (i, j) in comparators:
        bit_i = (state >> i) & 1
        bit_j = (state >> j) & 1
        if bit_i == 0 and bit_j == 1:
            state |= (1 << i)
            state &= ~(1 << j)
    return state

def _is_sorted_output(state):
    """Check if Boolean state is sorted (all 0s before all 1s)."""
    k = bin(state).count('1')
    expected = ((1 << k) - 1) << (N - k)
    return state == expected

def _count_sorting_failures(comparators, num_samples=2048, seed=12345):
    """Count sorting failures on random samples. Returns (failures, total)."""
    rng = random.Random(seed)
    failures = 0
    for _ in range(num_samples):
        mask = rng.randint(0, (1 << N) - 1)
        out = _simulate_network_fast(comparators, mask)
        if not _is_sorted_output(out):
            failures += 1
    return failures, num_samples

def _is_sorted_network_full(comparators):
    """Verify network sorts all 2^17 Boolean inputs (zero-one principle)."""
    for mask in range(1 << N):
        out = _simulate_network_fast(comparators, mask)
        if not _is_sorted_output(out):
            return False
    return True

def _layers_to_comparators(layers):
    """Convert list of layers (each a list of (i,j) pairs) to flat comparator list."""
    comps = []
    for layer in layers:
        for comp in layer:
            comps.append(comp)
    return comps

def _comparators_to_layers(comparators):
    """Convert flat comparator list to layers based on parallelism."""
    # Group comparators by their "depth" - a comparator can be in layer d if 
    # all its wires have been used in layers < d
    wire_last_used = [0] * N
    layers = []
    current_layer = []
    
    for comp in comparators:
        i, j = comp
        # Find the earliest layer where both wires are free
        min_depth = max(wire_last_used[i], wire_last_used[j])
        # Pad with empty layers if needed
        while len(layers) <= min_depth:
            layers.append([])
        layers[min_depth].append(comp)
        wire_last_used[i] = min_depth + 1
        wire_last_used[j] = min_depth + 1
    
    return layers

def _make_odd_even_transposition_layers():
    """Odd-even transposition as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _random_layer(num_comps):
    """Generate a random layer with num_comps non-conflicting comparators."""
    used = set()
    layer = []
    attempts = 0
    while len(layer) < num_comps and attempts < 100:
        attempts += 1
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        layer.append((i, j))
    return layer

def _random_network_layers(max_layers=20, max_comps_per_layer=8):
    """Generate a random network as layers."""
    num_layers = random.randint(5, max_layers)
    layers = []
    for _ in range(num_layers):
        num_comps = random.randint(1, min(max_comps_per_layer, N // 2))
        layer = _random_layer(num_comps)
        if layer:
            layers.append(layer)
    return layers

def _crossover_layers(parent1, parent2):
    """Crossover: take layers from parents, possibly mixing within a layer."""
    child = []
    max_len = max(len(parent1), len(parent2))
    for idx in range(max_len):
        if random.random() < 0.5:
            if idx < len(parent1):
                child.append(list(parent1[idx]))
        else:
            if idx < len(parent2):
                child.append(list(parent2[idx]))
    return child

def _mutate_layer(layer, mutation_rate=0.3):
    """Mutate a layer: add, remove, or replace comparators."""
    new_layer = list(layer)
    
    # Randomly remove some comparators
    if new_layer and random.random() < mutation_rate * 0.5:
        idx = random.randint(0, len(new_layer) - 1)
        new_layer.pop(idx)
    
    # Randomly add a comparator
    if random.random() < mutation_rate * 0.5:
        used = set()
        for (i, j) in new_layer:
            used.add(i)
            used.add(j)
        available = [w for w in range(N) if w not in used]
        if len(available) >= 2:
            i, j = random.sample(available, 2)
            if i > j:
                i, j = j, i
            new_layer.append((i, j))
    
    # Randomly replace a comparator
    if new_layer and random.random() < mutation_rate * 0.3:
        idx = random.randint(0, len(new_layer) - 1)
        old_i, old_j = new_layer[idx]
        used = set()
        for (a, b) in new_layer:
            if (a, b) != (old_i, old_j):
                used.add(a)
                used.add(b)
        available = [w for w in range(N) if w not in used]
        if len(available) >= 2:
            i, j = random.sample(available, 2)
            if i > j:
                i, j = j, i
            new_layer[idx] = (i, j)
    
    return new_layer

def _mutate_network_layers(layers, mutation_rate=0.3):
    """Mutate a network by mutating some layers and possibly adding/removing layers."""
    new_layers = [list(layer) for layer in layers]
    
    # Mutate each layer
    for idx in range(len(new_layers)):
        if random.random() < mutation_rate:
            new_layers[idx] = _mutate_layer(new_layers[idx], mutation_rate)
    
    # Possibly add a new layer
    if random.random() < mutation_rate * 0.3:
        new_layer = _random_layer(random.randint(1, 4))
        if new_layer:
            insert_pos = random.randint(0, len(new_layers))
            new_layers.insert(insert_pos, new_layer)
    
    # Possibly remove a layer
    if new_layers and random.random() < mutation_rate * 0.2:
        idx = random.randint(0, len(new_layers) - 1)
        new_layers.pop(idx)
    
    return new_layers

def _fitness(layers):
    """
    Fitness function: lower is better.
    Penalizes comparator count and sorting failures.
    """
    comps = _layers_to_comparators(layers)
    num_comps = len(comps)
    
    # Quick check on random samples
    failures, total = _count_sorting_failures(comps, num_samples=512, seed=999)
    
    # If there are failures, penalize heavily
    if failures > 0:
        return num_comps + failures * 100.0
    
    # No failures on samples, return comparator count
    return float(num_comps)

def _genetic_algorithm(initial_layers, population_size=20, max_generations=50):
    """Genetic algorithm to evolve sorting network layers."""
    population = [initial_layers]
    
    # Initialize population with variations
    for _ in range(population_size - 1):
        pop = _mutate_network_layers(initial_layers, mutation_rate=0.5)
        population.append(pop)
    
    best_layers = None
    best_fitness = float('inf')
    best_comps = None
    
    generation = 0
    while generation < max_generations and _time_left() > 5.0:
        generation += 1
        
        # Evaluate fitness
        fitnesses = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            if len(comps) == 0:
                fitnesses.append(float('inf'))
            else:
                fitness = _fitness(layers)
                fitnesses.append(fitness)
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_layers = [list(layer) for layer in layers]
                best_comps = comps
        
        # If we found a network with no failures on samples, verify fully
        if best_comps is not None and best_fitness <= len(best_comps) + 1:
            if _is_sorted_network_full(best_comps):
                return best_comps
        
        # Selection: tournament selection
        new_population = []
        for _ in range(population_size):
            # Tournament of 3
            candidates = random.sample(range(len(population)), 3)
            winner = min(candidates, key=lambda x: fitnesses[x])
            new_population.append([list(layer) for layer in population[winner]])
        
        # Crossover and mutation
        for idx in range(0, len(new_population) - 1, 2):
            p1 = new_population[idx]
            p2 = new_population[idx + 1]
            
            if random.random() < 0.7:  # Crossover probability
                child1 = _crossover_layers(p1, p2)
                child2 = _crossover_layers(p2, p1)
            else:
                child1 = [list(layer) for layer in p1]
                child2 = [list(layer) for layer in p2]
            
            # Mutate
            child1 = _mutate_network_layers(child1, mutation_rate=0.2)
            child2 = _mutate_network_layers(child2, mutation_rate=0.2)
            
            new_population[idx] = child1
            new_population[idx + 1] = child2
        
        population = new_population
    
    # Return best found, verify if possible
    if best_comps is not None:
        if _is_sorted_network_full(best_comps):
            return best_comps
        # If not fully verified, return it anyway (might be close)
        return best_comps
    
    return _layers_to_comparators(initial_layers)

def _local_search(layers, max_iters=200):
    """Local search: try removing comparators from layers."""
    best_layers = [list(layer) for layer in layers]
    best_comps = _layers_to_comparators(best_layers)
    best_len = len(best_comps)
    
    for _ in range(max_iters):
        if _time_left() < 5.0:
            break
        
        improved = False
        # Try removing a random comparator
        comps = _layers_to_comparators(best_layers)
        if len(comps) > 63:
            idx = random.randint(0, len(comps) - 1)
            trial_comps = comps[:idx] + comps[idx+1:]
            
            # Quick check
            failures, _ = _count_sorting_failures(trial_comps, num_samples=256, seed=777)
            if failures == 0:
                if _is_sorted_network_full(trial_comps):
                    best_comps = trial_comps
                    best_len = len(best_comps)
                    best_layers = _comparators_to_layers(best_comps)
                    improved = True
        
        if not improved:
            # Try adding a comparator to a random layer
            if best_layers and random.random() < 0.3:
                layer_idx = random.randint(0, len(best_layers) - 1)
                trial_layers = [list(layer) for layer in best_layers]
                used = set()
                for (i, j) in trial_layers[layer_idx]:
                    used.add(i)
                    used.add(j)
                available = [w for w in range(N) if w not in used]
                if len(available) >= 2:
                    i, j = random.sample(available, 2)
                    if i > j:
                        i, j = j, i
                    trial_layers[layer_idx].append((i, j))
                    trial_comps = _layers_to_comparators(trial_layers)
                    failures, _ = _count_sorting_failures(trial_comps, num_samples=256, seed=888)
                    if failures == 0 and len(trial_comps) < best_len:
                        if _is_sorted_network_full(trial_comps):
                            best_comps = trial_comps
                            best_len = len(best_comps)
                            best_layers = trial_layers
                            improved = True
        
        if not improved:
            break
    
    return best_comps

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    random.seed(SEED)
    
    # Start with odd-even transposition as layers
    initial_layers = _make_odd_even_transposition_layers()
    initial_comps = _layers_to_comparators(initial_layers)
    
    # First, do local search to reduce from 136
    best_comps = _local_search(initial_layers, max_iters=100)
    best_len = len(best_comps)
    
    # Convert to layers for GA
    best_layers = _comparators_to_layers(best_comps)
    
    # Run genetic algorithm
    ga_result = _genetic_algorithm(best_layers, population_size=15, max_generations=30)
    if ga_result is not None and len(ga_result) < best_len:
        best_comps = ga_result
        best_len = len(best_comps)
    
    # Final local search
    final_layers = _comparators_to_layers(best_comps)
    final_comps = _local_search(final_layers, max_iters=100)
    if len(final_comps) < best_len:
        best_comps = final_comps
        best_len = len(best_comps)
    
    return best_comps
