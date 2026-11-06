"""Rewritten program: Layered Genetic Algorithm for sorting networks on 17 wires."""
import numpy as np
import random
import sys
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55  # seconds, leaving buffer for the 60s limit

# Pre-compute the initial state for all 2^N boolean inputs
# This saves time in _verify_network
_NUM_INPUTS = 1 << N
_INITIAL_STATE = np.zeros((_NUM_INPUTS, N), dtype=np.uint8)
for j in range(N):
    _INITIAL_STATE[:, j] = ((np.arange(_NUM_INPUTS, dtype=np.uint32) >> j) & 1).astype(np.uint8)

def _verify_network(comparators):
    """
    Verify a sorting network using the Zero-One Principle with bit-vector simulation.
    comparators: list of (i, j) pairs.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    if not comparators:
        # Empty network only sorts if N <= 1, which is false for N=17
        return False
        
    # Copy the initial state
    state = _INITIAL_STATE.copy()
    
    for (i, j) in comparators:
        # Compare-exchange: min to i, max to j
        a = state[:, i]
        b = state[:, j]
        # min = a & b, max = a | b for boolean values
        # Using np.minimum and np.maximum is clear and fast enough
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    # Check if sorted: state[:, k] >= state[:, k+1] for all k
    # Equivalent to: no state[:, k] < state[:, k+1]
    # We check if any difference is negative
    diffs = state[:, :-1].astype(np.int8) - state[:, 1:].astype(np.int8)
    return np.all(diffs >= 0)

def _flatten_layers(layers):
    """Flatten a list of layers into a list of comparators."""
    comparators = []
    for layer in layers:
        comparators.extend(layer)
    return comparators

def _random_matching():
    """Generate a random matching (set of disjoint comparators) on N wires."""
    wires = list(range(N))
    random.shuffle(wires)
    matching = []
    for i in range(0, N - 1, 2):
        a, b = wires[i], wires[i+1]
        if a > b:
            a, b = b, a
        matching.append((a, b))
    return matching

def _random_network(num_layers=None):
    """Generate a random layered network."""
    if num_layers is None:
        num_layers = random.randint(5, 20)
    layers = []
    for _ in range(num_layers):
        layers.append(_random_matching())
    return layers

def _mutate_network(layers):
    """Mutate a layered network."""
    layers = [list(layer) for layer in layers]
    n_mutations = random.randint(1, 3)
    
    for _ in range(n_mutations):
        op = random.choice(['add_layer', 'remove_layer', 'mutate_layer', 'swap_layers'])
        
        if op == 'add_layer' and len(layers) < 50:
            pos = random.randint(0, len(layers))
            layers.insert(pos, _random_matching())
            
        elif op == 'remove_layer' and len(layers) > 1:
            pos = random.randint(0, len(layers) - 1)
            layers.pop(pos)
            
        elif op == 'mutate_layer' and layers:
            pos = random.randint(0, len(layers) - 1)
            # Replace the layer with a new random one
            layers[pos] = _random_matching()
                
        elif op == 'swap_layers' and len(layers) > 1:
            i, j = random.sample(range(len(layers)), 2)
            layers[i], layers[j] = layers[j], layers[i]
            
    return layers

def _crossover(parent1, parent2):
    """Crossover: combine layers from two parents."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    
    # Cut points
    cut1 = random.randint(0, len(parent1))
    cut2 = random.randint(0, len(parent2))
    
    child = parent1[:cut1] + parent2[cut2:]
    return [list(layer) for layer in child]

def _evaluate_fitness(layers):
    """
    Fitness: lower is better.
    If the network is valid, fitness = number of comparators.
    If invalid, fitness = a large penalty.
    """
    comparators = _flatten_layers(layers)
    if len(comparators) > MAX_COMPARATORS:
        return MAX_COMPARATORS + 1000
    if _verify_network(comparators):
        return len(comparators)
    else:
        return MAX_COMPARATORS + 1000

def _genetic_algorithm():
    """
    Run a genetic algorithm on layered networks.
    """
    start_time = time.time()
    
    pop_size = 40
    population = []
    
    # Initialize with seed (odd-even transposition)
    # Odd-even transposition:
    # Phase 0: (0,1), (2,3), ...
    # Phase 1: (1,2), (3,4), ...
    seed_layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        seed_layers.append(layer)
    population.append(seed_layers)
    
    # Add random networks
    for _ in range(pop_size - 1):
        population.append(_random_network())
    
    # Evaluate initial population
    fitnesses = [_evaluate_fitness(ind) for ind in population]
    
    best_network = min(zip(fitnesses, population), key=lambda x: x[0])
    best_fitness = best_network[0]
    best_comparators = _flatten_layers(best_network[1])
    
    print(f"Initial best: {best_fitness} comparators", file=sys.stderr)
    
    generation = 0
    while time.time() - start_time < TIME_LIMIT:
        generation += 1
        
        # Sort population by fitness
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        
        # Elitism
        elite_size = max(2, pop_size // 5)
        new_population = [ind for _, ind in sorted_pop[:elite_size]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            def select_tournament():
                p1 = random.choice(sorted_pop[:elite_size])[1]
                p2 = random.choice(sorted_pop[:elite_size])[1]
                return p1, p2
            
            p1, p2 = select_tournament()
            
            if random.random() < 0.5:
                child = _crossover(p1, p2)
            else:
                child = _mutate_network(p1)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_evaluate_fitness(ind) for ind in population]
        
        # Update best
        gen_best = min(zip(fitnesses, population), key=lambda x: x[0])
        if gen_best[0] < best_fitness:
            best_fitness = gen_best[0]
            best_comparators = _flatten_layers(gen_best[1])
            print(f"Gen {generation}: new best {best_fitness} comparators", file=sys.stderr)
            
            # Local search on the best: try to remove redundant comparators
            if best_fitness < 136:
                improved = True
                while improved:
                    improved = False
                    for i in range(len(best_comparators) - 1, -1, -1):
                        candidate = best_comparators[:i] + best_comparators[i+1:]
                        if _verify_network(candidate):
                            best_comparators = candidate
                            best_fitness = len(candidate)
                            improved = True
                            break
                print(f"Local search: new best {best_fitness} comparators", file=sys.stderr)
    
    return best_comparators

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    try:
        network = _genetic_algorithm()
        # Verify the result
        if _verify_network(network):
            return network
    except Exception as e:
        print(f"GA failed: {e}", file=sys.stderr)
    
    # Fallback to odd-even transposition
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
