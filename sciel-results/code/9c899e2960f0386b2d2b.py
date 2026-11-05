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

def _evaluate_network(comparators, samples=None, sample_size=2048):
    """
    Evaluate a sorting network using the zero-one principle with random Boolean inputs.
    Returns True if the network sorts all sampled inputs.
    """
    if samples is None:
        samples = np.random.randint(0, 2, size=(sample_size, N))
    
    for sample in samples:
        state = sample.copy()
        for i, j in comparators:
            _apply_comparator(state, i, j)
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
    level = [0] * N
    depth = 0
    for i, j in comparators:
        lvl = max(level[i], level[j]) + 1
        level[i] = lvl
        level[j] = lvl
        depth = max(depth, lvl)
    return depth

def _generate_seed_network():
    """Generate the odd-even transposition network as seed."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _layers_to_comparators(layers):
    """Convert a list of layers (each a list of (i,j) pairs) to a flat list of comparators."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps

def _comparators_to_layers(comparators):
    """Convert a flat list of comparators to layers using greedy assignment."""
    if not comparators:
        return []
    
    # Assign each comparator to the earliest possible layer
    # where its wires are free
    wire_free_at = [0] * N  # wire_free_at[w] = earliest layer index where wire w is free
    layers = []
    
    for comp in comparators:
        i, j = comp
        # Find the earliest layer where both wires are free
        target_layer = max(wire_free_at[i], wire_free_at[j])
        
        # Ensure this layer exists
        while len(layers) <= target_layer:
            layers.append([])
        
        # Check if the layer already has a comparator using i or j
        # (shouldn't happen with greedy, but safety check)
        for existing_i, existing_j in layers[target_layer]:
            if existing_i == i or existing_i == j or existing_j == i or existing_j == j:
                # Conflict - move to next layer
                target_layer += 1
                while len(layers) <= target_layer:
                    layers.append([])
        
        layers[target_layer].append((i, j))
        wire_free_at[i] = target_layer + 1
        wire_free_at[j] = target_layer + 1
    
    return [layer for layer in layers if layer]

def _make_random_layer():
    """Create a random maximum matching layer for N wires."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    for k in range(0, N - 1, 2):
        i, j = wires[k], wires[k + 1]
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

def _make_random_network(num_layers=None):
    """Create a random sorting network with random layers."""
    if num_layers is None:
        num_layers = random.randint(10, 20)
    layers = [_make_random_layer() for _ in range(num_layers)]
    return layers

def _mutate_layers(layers, mutation_rate=0.15):
    """
    Mutate a layer-based network:
    - Swap comparators within a layer
    - Swap comparators between layers
    - Remove a random comparator from a layer
    - Add a random comparator to a layer (if wire is free)
    - Shuffle wires in a random layer
    """
    new_layers = [list(layer) for layer in layers]
    n = len(new_layers)
    
    num_mutations = max(1, int(mutation_rate * n * 3))
    
    for _ in range(num_mutations):
        op = random.choice(['swap_within', 'swap_between', 'remove', 'add', 'shuffle_layer'])
        
        if op == 'swap_within' and n > 0:
            # Swap two comparators within the same layer
            layer_idx = random.randint(0, n - 1)
            if len(new_layers[layer_idx]) >= 2:
                a, b = random.sample(range(len(new_layers[layer_idx])), 2)
                new_layers[layer_idx][a], new_layers[layer_idx][b] = \
                    new_layers[layer_idx][b], new_layers[layer_idx][a]
        
        elif op == 'swap_between' and n >= 2:
            # Swap a comparator from one layer with one from another
            l1, l2 = random.sample(range(n), 2)
            if new_layers[l1] and new_layers[l2]:
                idx1 = random.randint(0, len(new_layers[l1]) - 1)
                idx2 = random.randint(0, len(new_layers[l2]) - 1)
                comp1 = new_layers[l1][idx1]
                comp2 = new_layers[l2][idx2]
                # Check if swap is valid (no wire conflicts in either layer)
                valid = True
                for other in new_layers[l1]:
                    if other != comp1 and (other[0] in comp2 or other[1] in comp2):
                        valid = False
                        break
                if valid:
                    for other in new_layers[l2]:
                        if other != comp2 and (other[0] in comp1 or other[1] in comp1):
                            valid = False
                            break
                if valid:
                    new_layers[l1][idx1] = comp2
                    new_layers[l2][idx2] = comp1
        
        elif op == 'remove' and n > 0:
            # Remove a random comparator from a random layer
            layer_idx = random.randint(0, n - 1)
            if new_layers[layer_idx]:
                comp_idx = random.randint(0, len(new_layers[layer_idx]) - 1)
                new_layers[layer_idx].pop(comp_idx)
            # Remove empty layers
            new_layers = [layer for layer in new_layers if layer]
        
        elif op == 'add' and n > 0:
            # Add a random comparator to a random layer if wires are free
            layer_idx = random.randint(0, n - 1)
            used_wires = set()
            for i, j in new_layers[layer_idx]:
                used_wires.add(i)
                used_wires.add(j)
            free_wires = [w for w in range(N) if w not in used_wires]
            if len(free_wires) >= 2:
                i, j = random.sample(free_wires, 2)
                if i > j:
                    i, j = j, i
                new_layers[layer_idx].append((i, j))
        
        elif op == 'shuffle_layer' and n > 0:
            # Re-shuffle a random layer's matching
            layer_idx = random.randint(0, n - 1)
            # Get all wires used in this layer
            wires_in_layer = set()
            for i, j in new_layers[layer_idx]:
                wires_in_layer.add(i)
                wires_in_layer.add(j)
            wires_list = list(wires_in_layer)
            random.shuffle(wires_list)
            new_layer = []
            for k in range(0, len(wires_list) - 1, 2):
                i, j = wires_list[k], wires_list[k + 1]
                if i > j:
                    i, j = j, i
                new_layer.append((i, j))
            new_layers[layer_idx] = new_layer
    
    return new_layers

def _crossover_layers(parent1, parent2):
    """
    Crossover: take a random prefix of layers from parent1 and suffix from parent2.
    """
    if not parent1:
        return [list(layer) for layer in parent2]
    if not parent2:
        return [list(layer) for layer in parent1]
    
    max_len = max(len(parent1), len(parent2))
    split = random.randint(0, max_len)
    
    child = []
    for i in range(max_len):
        if i < split and i < len(parent1):
            child.append(list(parent1[i]))
        elif i < len(parent2):
            child.append(list(parent2[i]))
        elif i < len(parent1):
            child.append(list(parent1[i]))
    
    # Remove empty layers
    child = [layer for layer in child if layer]
    return child

def _local_search_layers(layers, max_iters=30, sample_size=512):
    """
    Local search: try removing comparators from layers that are redundant.
    """
    comps = _layers_to_comparators(layers)
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        for idx in range(len(comps) - 1, -1, -1):
            candidate = comps[:idx] + comps[idx+1:]
            if _evaluate_network(candidate, sample_size=sample_size):
                comps = candidate
                changed = True
                break
    # Convert back to layers
    return _comparators_to_layers(comps)

def _count_comparators(layers):
    """Count total comparators in a layer-based network."""
    return sum(len(layer) for layer in layers)

def construct():
    """
    Construct a sorting network using a genetic algorithm with layer-based representation.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    max_time = 55  # Leave some buffer
    
    # Initialize with seed network converted to layers
    seed_comps = _generate_seed_network()
    best_layers = _comparators_to_layers(seed_comps)
    best_count = len(seed_comps)
    
    # Population of layer-based networks
    pop_size = 25
    population = [best_layers]
    for _ in range(pop_size - 1):
        mutated = _mutate_layers(best_layers, mutation_rate=0.2)
        population.append(mutated)
    
    # Pre-generate samples for fast evaluation
    np.random.seed(42)
    samples = np.random.randint(0, 2, size=(1024, N))
    
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate and select
        scored = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            # Quick evaluation with random samples
            if _evaluate_network(comps, samples=samples, sample_size=1024):
                score = -len(comps)
            else:
                score = -10000
            scored.append((score, layers))
        
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] > -10000:
            best_comps_candidate = _layers_to_comparators(scored[0][1])
            if len(best_comps_candidate) < best_count:
                if _evaluate_network_full(best_comps_candidate):
                    best_layers = scored[0][1]
                    best_count = len(best_comps_candidate)
        
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
                parent2 = candidates[1][1] if len(candidates) > 1 else best
            
            child = _crossover_layers(parent1, parent2)
            child = _mutate_layers(child, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
        
        # Local search on best candidate occasionally
        if generation % 5 == 0 and time.time() - start_time < max_time - 10:
            local_result = _local_search_layers(best_layers, max_iters=20, sample_size=512)
            local_comps = _layers_to_comparators(local_result)
            if len(local_comps) < best_count:
                if _evaluate_network_full(local_comps):
                    best_layers = local_result
                    best_count = len(local_comps)
    
    return _layers_to_comparators(best_layers)

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
