"""
Layer-based Genetic Algorithm for Sorting Network Optimization.
Represents the network as a sequence of parallel comparison layers.
"""
import numpy as np
import time
import random

N = 17
DEADLINE = 85  # safety margin before 90s kill
SAMPLE_SIZE = 4096  # Use 4096 random inputs for fast validation

# Precompute sample inputs
np.random.seed(42)
_SAMPLE_INPUTS = np.random.randint(0, 2**N, size=SAMPLE_SIZE, dtype=np.uint32)

def _simulate_network_fast(network, inputs):
    """Faster simulation using vectorized numpy operations."""
    wires = [np.empty(len(inputs), dtype=np.uint8) for _ in range(N)]
    for i in range(N):
        wires[i] = (inputs >> i) & 1
    
    for (a, b) in network:
        wa = wires[a]
        wb = wires[b]
        wires[a] = wa & wb
        wires[b] = wa | wb
    
    # Check sorted
    for i in range(N - 1):
        if np.any((wires[i] == 1) & (wires[i+1] == 0)):
            return False
    return True

def _is_sorted_network_sample(network):
    """Check if network sorts all sample boolean inputs (fast)."""
    return _simulate_network_fast(network, _SAMPLE_INPUTS)

def _is_sorted_network_full(network):
    """Check if network sorts all 131072 boolean inputs (slow but exact)."""
    all_inputs = np.arange(2**N, dtype=np.uint32)
    return _simulate_network_fast(network, all_inputs)

def _layers_to_flat(layers):
    """Convert list of layers (each a list of (i,j) pairs) to flat list."""
    flat = []
    for layer in layers:
        flat.extend(layer)
    return flat

def _flat_to_layers(flat):
    """Convert flat list to list of layers, assuming comparators in the same layer are disjoint.
    This is a heuristic conversion: we assume the input flat list was generated from layers.
    However, for GA, we operate directly on layers."""
    # This function is not strictly needed if we start with layers and evolve layers.
    # But useful for initializing from a flat network if needed.
    # For now, we will initialize layers manually or from a known structure.
    pass

def _create_initial_layers():
    """Create an initial set of layers based on Odd-Even Merge Sort or similar."""
    # We will use a simple construction: 
    # Odd-even transposition sort has N layers.
    # But we want to start with a valid network.
    # Let's use the odd-even merge sort network structure if we can generate it as layers.
    # Alternatively, start with a dense random valid network? No, too slow to validate.
    # Let's start with the seed: odd-even transposition.
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _layer_is_disjoint(layer):
    """Check if a layer contains only disjoint comparators."""
    used = set()
    for i, j in layer:
        if i in used or j in used:
            return False
        used.add(i)
        used.add(j)
    return True

def _mutate_layer(layer, mutation_rate=0.1):
    """Mutate a single layer."""
    new_layer = layer[:]
    
    # 1. Add a comparator
    if random.random() < mutation_rate and len(new_layer) < N // 2:
        # Find available wires
        used = set()
        for i, j in new_layer:
            used.add(i)
            used.add(j)
        available = [i for i in range(N) if i not in used]
        if len(available) >= 2:
            a, b = random.sample(available, 2)
            if a > b: a, b = b, a
            new_layer.append((a, b))
            
    # 2. Remove a comparator
    elif random.random() < mutation_rate and len(new_layer) > 0:
        idx = random.randint(0, len(new_layer) - 1)
        new_layer.pop(idx)
        
    # 3. Swap a wire in a comparator
    elif random.random() < mutation_rate * 0.5 and len(new_layer) > 0:
        idx = random.randint(0, len(new_layer) - 1)
        i, j = new_layer[idx]
        # Try to swap i with another wire
        used = set()
        for k, l in new_layer:
            if (k, l) != (i, j) and (l, k) != (i, j):
                used.add(k)
                used.add(l)
        # i is free to change? No, i is used in this comparator.
        # We can try to change i to another unused wire.
        available = [x for x in range(N) if x not in used and x != j]
        if available:
            new_i = random.choice(available)
            if new_i < j:
                new_layer[idx] = (new_i, j)
            else:
                new_layer[idx] = (j, new_i)
                
    return new_layer

def _crossover_layers(parent1, parent2):
    """Crossover two layer structures."""
    # Uniform crossover on layers: for each layer index, pick from parent1 or parent2
    # But layers have different lengths.
    # Strategy: Align by index. If one is shorter, pad with empty layers?
    # Or, just pick a random number of layers from the front of each?
    # Let's try: Pick a random cut point for each parent and combine.
    
    max_len = max(len(parent1), len(parent2))
    child_layers = []
    
    for i in range(max_len):
        if random.random() < 0.5:
            if i < len(parent1):
                child_layers.append(parent1[i][:])
            else:
                child_layers.append([])
        else:
            if i < len(parent2):
                child_layers.append(parent2[i][:])
            else:
                child_layers.append([])
                
    # Remove empty layers at the end
    while child_layers and not child_layers[-1]:
        child_layers.pop()
        
    return child_layers

def _fitness(layers):
    """Calculate fitness: lower is better.
    Primary: Number of comparators.
    Penalty: If not sorted on sample, add a large penalty."""
    flat = _layers_to_flat(layers)
    num_comp = len(flat)
    
    if not _is_sorted_network_sample(flat):
        return num_comp + 1000  # Large penalty
    
    return num_comp

def _ga_evolve(initial_layers, time_budget):
    """Run Genetic Algorithm to evolve layers."""
    start_time = time.time()
    
    # Initialize population
    pop_size = 20
    population = [initial_layers]
    
    # Add some variations
    for _ in range(pop_size - 1):
        mutated = []
        for layer in initial_layers:
            mutated.append(_mutate_layer(layer, mutation_rate=0.3))
        population.append(mutated)
        
    # Evaluate initial fitness
    fitnesses = [_fitness(ind) for ind in population]
    
    best_layers = population[0]
    best_fitness = fitnesses[0]
    
    for gen in range(100): # Max generations
        if time.time() - start_time > time_budget:
            break
            
        # Selection: Tournament
        selected = []
        for _ in range(pop_size):
            candidates = random.sample(list(zip(population, fitnesses)), 3)
            best_candidate = min(candidates, key=lambda x: x[1])
            selected.append(best_candidate[0])
            
        # Crossover and Mutation
        new_population = []
        for i in range(0, pop_size - 1, 2):
            p1 = selected[i]
            p2 = selected[i+1]
            
            child = _crossover_layers(p1, p2)
            
            # Mutate child
            for j in range(len(child)):
                child[j] = _mutate_layer(child[j], mutation_rate=0.1)
                
            # Ensure no empty layers at end
            while child and not child[-1]:
                child.pop()
                
            new_population.append(child)
            
            # Elitism: keep one parent
            new_population.append(p1)
            
        # Replace population
        population = new_population
        fitnesses = [_fitness(ind) for ind in population]
        
        # Update best
        for i, fit in enumerate(fitnesses):
            if fit < best_fitness:
                best_fitness = fit
                best_layers = population[i]
                
    return best_layers

def construct():
    """Construct a sorting network using layer-based GA."""
    start_time = time.time()
    time_budget = DEADLINE - (time.time() - start_time) - 5
    
    if time_budget < 10:
        # Not enough time, return seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Start with odd-even transposition layers
    initial_layers = _create_initial_layers()
    
    # Run GA
    best_layers = _ga_evolve(initial_layers, time_budget * 0.9)
    
    # Convert to flat
    best_flat = _layers_to_flat(best_layers)
    
    # Final verification
    if _is_sorted_network_full(best_flat):
        return best_flat
    else:
        # Fallback to seed
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
