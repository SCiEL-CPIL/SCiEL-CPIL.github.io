"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 17

def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a state vector (in-place)."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _flatten_layers(layers):
    """Flatten a list of layers into a single list of comparators."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps

def _evaluate_network(comparators, samples=None, sample_size=2048):
    """
    Evaluate a sorting network using the zero-one principle with random Boolean inputs.
    Returns True if the network sorts all sampled inputs.
    """
    if samples is None:
        # Generate random Boolean inputs
        samples = np.random.randint(0, 2, size=(sample_size, N))
    
    for sample in samples:
        state = sample.copy()
        for i, j in comparators:
            _apply_comparator(state, i, j)
        # Check if sorted
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _evaluate_network_full(comparators):
    """
    Evaluate a sorting network over ALL 2^17 Boolean inputs (zero-one principle).
    Returns True if the network sorts all inputs.
    """
    for mask in range(1 << N):
        state = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        for i, j in comparators:
            _apply_comparator(state, i, j)
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _compute_depth(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    # Assign each comparator to a layer (level)
    # A comparator can be at level max(level[i], level[j]) + 1
    level = [0] * N
    depth = 0
    for i, j in comparators:
        lvl = max(level[i], level[j]) + 1
        level[i] = lvl
        level[j] = lvl
        depth = max(depth, lvl)
    return depth

def _generate_seed_layers():
    """Generate the odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        if layer:
            layers.append(layer)
    return layers

def _is_valid_layer(layer):
    """Check if a layer is a valid matching (no shared wires)."""
    wires = set()
    for i, j in layer:
        if i in wires or j in wires:
            return False
        wires.add(i)
        wires.add(j)
    return True

def _mutate_layers(layers, mutation_rate=0.1):
    """
    Apply mutations to the layer structure.
    """
    new_layers = [list(layer) for layer in layers]
    n_layers = len(new_layers)
    
    # Number of mutations
    n_muts = max(1, int(mutation_rate * (n_layers + N)))
    
    for _ in range(n_muts):
        op = random.choice(['add_layer', 'remove_layer', 'modify_layer', 'swap_layers'])
        
        if op == 'add_layer' and n_layers < 50:
            # Add a random matching layer
            new_layer = []
            available = list(range(N))
            random.shuffle(available)
            while len(available) >= 2:
                i = available.pop()
                j = available.pop()
                if i > j: i, j = j, i
                new_layer.append((i, j))
            new_layers.append(new_layer)
            
        elif op == 'remove_layer' and n_layers > 1:
            idx = random.randint(0, n_layers - 1)
            new_layers.pop(idx)
            
        elif op == 'modify_layer' and n_layers > 0:
            idx = random.randint(0, n_layers - 1)
            layer = new_layers[idx]
            # Try to modify a comparator in the layer
            if layer:
                c_idx = random.randint(0, len(layer) - 1)
                i, j = layer[c_idx]
                # Option 1: Remove comparator
                if random.random() < 0.3:
                    layer.pop(c_idx)
                else:
                    # Option 2: Change one wire
                    new_i = i
                    new_j = j
                    if random.random() < 0.5:
                        # Change i
                        candidates = [w for w in range(N) if w != i and w != j]
                        if candidates:
                            new_i = random.choice(candidates)
                    else:
                        # Change j
                        candidates = [w for w in range(N) if w != i and w != j]
                        if candidates:
                            new_j = random.choice(candidates)
                    
                    if new_i > new_j:
                        new_i, new_j = new_j, new_i
                    
                    # Check validity with rest of layer
                    rest_wires = set()
                    for k, l in layer:
                        if (k, l) != (i, j):
                            rest_wires.add(k)
                            rest_wires.add(l)
                    
                    if new_i not in rest_wires and new_j not in rest_wires:
                        layer[c_idx] = (new_i, new_j)
            else:
                # Add a comparator to empty layer
                i, j = random.sample(range(N), 2)
                if i > j: i, j = j, i
                layer.append((i, j))
                
        elif op == 'swap_layers' and n_layers > 1:
            i, j = random.sample(range(n_layers), 2)
            new_layers[i], new_layers[j] = new_layers[j], new_layers[i]
            
    return new_layers

def _crossover_layers(parent1, parent2):
    """Crossover: take prefix from one, suffix from other."""
    if not parent1 or not parent2:
        return [list(l) for l in (parent1 or parent2)]
    
    len1 = len(parent1)
    len2 = len(parent2)
    max_len = max(len1, len2)
    split = random.randint(0, max_len)
    
    child = []
    for i in range(max_len):
        if i < split:
            if i < len1:
                child.append(list(parent1[i]))
            else:
                child.append([])
        else:
            if i < len2:
                child.append(list(parent2[i]))
            else:
                child.append([])
                
    # Remove empty layers
    child = [l for l in child if l]
    return child

def _local_search_layers(layers, max_iters=20):
    """
    Local search: try to remove redundant comparators or merge layers.
    """
    current_layers = [list(l) for l in layers]
    current_comps = _flatten_layers(current_layers)
    current_len = len(current_comps)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Try removing a comparator from each layer
        for l_idx in range(len(current_layers)):
            layer = current_layers[l_idx]
            for c_idx in range(len(layer) - 1, -1, -1):
                # Remove comparator
                new_layer = layer[:c_idx] + layer[c_idx+1:]
                new_layers = [list(l) for l in current_layers]
                new_layers[l_idx] = new_layer
                
                new_comps = _flatten_layers(new_layers)
                if len(new_comps) < current_len:
                    # Quick check
                    if _evaluate_network(new_comps, sample_size=512):
                        # Full check
                        if _evaluate_network_full(new_comps):
                            current_layers = new_layers
                            current_comps = new_comps
                            current_len = len(new_comps)
                            improved = True
                            break
            if improved:
                break
                
    return current_layers

def construct():
    """
    Construct a sorting network using a genetic algorithm with layer representation.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    max_time = 55  # Leave some buffer
    
    # Initialize with seed network
    best_layers = _generate_seed_layers()
    best_comps = _flatten_layers(best_layers)
    best_len = len(best_comps)
    
    # Population
    pop_size = 30
    population = [best_layers]
    for _ in range(pop_size - 1):
        mutated = _mutate_layers(best_layers, mutation_rate=0.2)
        population.append(mutated)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate and select
        scored = []
        for layers in population:
            comps = _flatten_layers(layers)
            # Quick evaluation with random samples
            if _evaluate_network(comps, sample_size=1024):
                score = -len(comps)  # Lower is better (we maximize, so negate)
            else:
                score = -10000
            scored.append((score, layers, comps))
        
        scored.sort(key=lambda x: x[0])  # Best first (most negative = fewest comparators)
        
        # Update best
        if scored[0][0] > -10000:  # Valid
            best_candidate_layers = scored[0][1]
            best_candidate_comps = scored[0][2]
            if len(best_candidate_comps) < best_len:
                # Full verification
                if _evaluate_network_full(best_candidate_comps):
                    best_layers = best_candidate_layers
                    best_comps = best_candidate_comps
                    best_len = len(best_comps)
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best few
        for i in range(min(5, len(scored))):
            if scored[i][0] > -10000:
                new_population.append(scored[i][1])
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            if len(new_population) < 2:
                parent1 = best_layers
                parent2 = best_layers
            else:
                # Tournament selection
                candidates = random.sample(scored, min(3, len(scored)))
                candidates.sort(key=lambda x: x[0])
                parent1 = candidates[0][1]
                parent2 = candidates[1][1] if len(candidates) > 1 else best_layers
            
            child = _crossover_layers(parent1, parent2)
            child = _mutate_layers(child, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
        
        # Local search on best candidate occasionally
        if generation % 5 == 0 and time.time() - start_time < max_time - 10:
            local_result = _local_search_layers(best_layers, max_iters=20)
            local_comps = _flatten_layers(local_result)
            if len(local_comps) < best_len:
                if _evaluate_network_full(local_comps):
                    best_layers = local_result
                    best_comps = local_comps
                    best_len = len(best_comps)
    
    return best_comps

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
