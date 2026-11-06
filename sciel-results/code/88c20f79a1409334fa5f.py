import time
import random

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 50.0  # Safety margin from 60s limit

# Pre-compute masks at module load time to avoid doing it in construct()
_INITIAL_MASKS = [0] * N
for _w in range(N):
    block = 1 << _w
    pattern = (1 << block) - 1
    pattern <<= block
    num_periods = (1 << N) // (2 * block)
    m = 0
    shift = 0
    for _ in range(num_periods):
        m |= (pattern << shift)
        shift += 2 * block
    _INITIAL_MASKS[_w] = m

def _odd_even_transposition(n=N):
    """Return the odd-even transposition sorting network for n wires."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def _verify(comparators):
    """
    Verify if the comparator list sorts all 2**N boolean inputs.
    Uses bitwise operations on Python integers representing the state of each wire.
    """
    if not comparators:
        return N <= 1
    
    state = _INITIAL_MASKS[:]
    
    for i, j in comparators:
        a = state[i]
        b = state[j]
        state[i] = a & b
        state[j] = a | b
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _verify_layered(layers):
    """
    Verify a layered network. Each layer is a list of (i,j) pairs that can run in parallel.
    """
    if not layers:
        return N <= 1
    
    state = _INITIAL_MASKS[:]
    
    for layer in layers:
        new_state = state[:]
        for i, j in layer:
            a = state[i]
            b = state[j]
            new_state[i] = a & b
            new_state[j] = a | b
        state = new_state
        
    for i in range(N - 1):
        if (state[i] & ~state[i+1]) != 0:
            return False
            
    return True

def _layers_to_comparators(layers):
    """Convert layered representation to flat comparator list."""
    comparators = []
    for layer in layers:
        comparators.extend(layer)
    return comparators

def _comparators_to_layers(comparators):
    """
    Convert a flat comparator list to layers by grouping comparators that don't conflict.
    A conflict occurs when two comparators share a wire.
    """
    if not comparators:
        return []
    
    layers = []
    current_layer = []
    used_wires = set()
    
    for comp in comparators:
        i, j = comp
        if i not in used_wires and j not in used_wires:
            current_layer.append(comp)
            used_wires.add(i)
            used_wires.add(j)
        else:
            if current_layer:
                layers.append(current_layer)
            current_layer = [comp]
            used_wires = {i, j}
    
    if current_layer:
        layers.append(current_layer)
    
    return layers

def _random_layer(n_wires=N, max_comps_per_layer=9, rng=None):
    """Generate a random valid layer (no wire appears twice in a layer)."""
    if rng is None:
        rng = random
    
    wires = list(range(n_wires))
    rng.shuffle(wires)
    
    layer = []
    i = 0
    while i < len(wires) - 1 and len(layer) < max_comps_per_layer:
        a = wires[i]
        b = wires[i + 1]
        if a < b:
            layer.append((a, b))
        else:
            layer.append((b, a))
        i += 2
    
    return layer

def _mutate_layer(layer, rng, mutation_rate=0.3):
    """Mutate a single layer by randomly changing comparators."""
    if not layer:
        return layer
    
    new_layer = []
    for comp in layer:
        if rng.random() < mutation_rate:
            # Replace with a random valid comparator
            a = rng.randint(0, N - 2)
            b = rng.randint(a + 1, N - 1)
            new_layer.append((a, b))
        else:
            new_layer.append(comp)
    
    # Ensure no wire is used twice
    seen = set()
    valid_layer = []
    for comp in new_layer:
        i, j = comp
        if i not in seen and j not in seen:
            valid_layer.append(comp)
            seen.add(i)
            seen.add(j)
    
    return valid_layer

def _crossover_layers(layers1, layers2, rng):
    """
    Crossover between two layered networks.
    We mix layers from both parents.
    """
    if not layers1 or not layers2:
        return layers1 if layers1 else layers2
    
    # Decide how many layers to take from each parent
    n1 = len(layers1)
    n2 = len(layers2)
    
    # Randomly decide the number of layers in offspring
    n_offspring = rng.randint(min(n1, n2), max(n1, n2))
    
    offspring = []
    for k in range(n_offspring):
        # Choose which parent to take this layer from
        if rng.random() < 0.5:
            if k < n1:
                layer = layers1[k]
            else:
                layer = layers1[rng.randint(0, n1 - 1)]
        else:
            if k < n2:
                layer = layers2[k]
            else:
                layer = layers2[rng.randint(0, n2 - 1)]
        
        # Mutate the layer
        layer = _mutate_layer(layer, rng, mutation_rate=0.2)
        offspring.append(layer)
    
    return offspring

def _greedy_reduce_layered(layers, deadline, max_iterations=50):
    """
    Greedily remove comparators from a layered network.
    Try removing each comparator and keep it if the network remains valid.
    """
    current = [layer[:] for layer in layers]
    
    for _ in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Try removing a random comparator from a random layer
        if not current or all(len(l) == 0 for l in current):
            break
        
        layer_idx = random.randint(0, len(current) - 1)
        if not current[layer_idx]:
            continue
        
        comp_idx = random.randint(0, len(current[layer_idx]) - 1)
        candidate = [l[:] for l in current]
        candidate[layer_idx].pop(comp_idx)
        
        if _verify_layered(candidate):
            current = candidate
            # Remove empty layers
            current = [l for l in current if l]
        else:
            # Try a different random removal
            pass
    
    return current

def _fitness(layers):
    """
    Fitness function for a layered network.
    Lower is better: total number of comparators, with penalty for invalid networks.
    """
    if not _verify_layered(layers):
        return float('inf')
    
    total_comps = sum(len(layer) for layer in layers)
    return total_comps

def _genetic_algorithm_layered(initial_layers, deadline, population_size=20, generations=100):
    """
    Genetic algorithm that evolves layered sorting networks.
    """
    rng = random.Random(42)
    
    # Initialize population
    population = [initial_layers]
    
    # Add diverse random individuals
    for _ in range(population_size - 1):
        n_layers = rng.randint(8, 15)
        layers = []
        for _ in range(n_layers):
            layer = _random_layer(N, max_comps_per_layer=9, rng=rng)
            layers.append(layer)
        population.append(layers)
    
    # Evaluate initial population
    fitnesses = [_fitness(ind) for ind in population]
    
    # Track best
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_individual = [layer[:] for layer in population[best_idx]]
    best_fitness = fitnesses[best_idx]
    
    for gen in range(generations):
        if time.time() > deadline:
            break
        
        # Sort by fitness
        sorted_indices = sorted(range(len(population)), key=lambda i: fitnesses[i])
        
        # Elitism: keep the best individual
        new_population = [[layer[:] for layer in population[sorted_indices[0]]]]
        new_fitnesses = [fitnesses[sorted_indices[0]]]
        
        # Generate offspring
        while len(new_population) < population_size:
            if time.time() > deadline:
                break
            
            # Select parents using tournament selection
            def tournament_select(k=3):
                candidates = rng.sample(range(len(population)), k)
                best_cand = min(candidates, key=lambda i: fitnesses[i])
                return population[best_cand]
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            child = _crossover_layers(parent1, parent2, rng)
            
            # Mutation: occasionally add or remove a layer
            if rng.random() < 0.1 and child:
                # Add a random layer
                new_layer = _random_layer(N, max_comps_per_layer=6, rng=rng)
                if new_layer:
                    insert_pos = rng.randint(0, len(child))
                    child.insert(insert_pos, new_layer)
            elif rng.random() < 0.1 and len(child) > 5:
                # Remove a random layer
                remove_idx = rng.randint(0, len(child) - 1)
                child.pop(remove_idx)
            
            # Evaluate child
            child_fitness = _fitness(child)
            
            # Greedy reduction on good children
            if child_fitness < best_fitness:
                child = _greedy_reduce_layered(child, deadline, max_iterations=10)
                child_fitness = _fitness(child)
            
            new_population.append(child)
            new_fitnesses.append(child_fitness)
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Update best
        best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
        if fitnesses[best_idx] < best_fitness:
            best_fitness = fitnesses[best_idx]
            best_individual = [layer[:] for layer in population[best_idx]]
    
    return best_individual

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS

    # Build initial network using odd-even transposition
    initial_comparators = _odd_even_transposition()
    
    # Convert to layers
    initial_layers = _comparators_to_layers(initial_comparators)
    
    # Phase 1: Greedy reduction on the layered network
    current_layers = _greedy_reduce_layered(initial_layers, deadline, max_iterations=30)
    
    # Convert to flat list and verify
    current_comparators = _layers_to_comparators(current_layers)
    if not _verify(current_comparators):
        current_comparators = initial_comparators[:]
        current_layers = _comparators_to_layers(current_comparators)
    
    best_comparators = current_comparators[:]
    best_len = len(best_comparators)
    
    # Phase 2: Genetic Algorithm on layered representation
    if time.time() < deadline - 5:
        ga_layers = _genetic_algorithm_layered(current_layers, deadline, 
                                               population_size=15, generations=50)
        ga_comparators = _layers_to_comparators(ga_layers)
        
        if _verify(ga_comparators) and len(ga_comparators) < best_len:
            best_comparators = ga_comparators
            best_len = len(best_comparators)
    
    # Phase 3: Final greedy reduction on the best solution
    if time.time() < deadline - 2:
        final_layers = _comparators_to_layers(best_comparators)
        final_layers = _greedy_reduce_layered(final_layers, deadline, max_iterations=50)
        final_comparators = _layers_to_comparators(final_layers)
        
        if _verify(final_comparators) and len(final_comparators) < best_len:
            best_comparators = final_comparators
            best_len = len(best_comparators)
    
    # Final verification
    if not _verify(best_comparators):
        return _odd_even_transposition()
        
    return best_comparators
