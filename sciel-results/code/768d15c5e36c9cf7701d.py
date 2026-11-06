"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55.0  # Leave some buffer for 60s limit

# EVOLVE-BLOCK-START

def _apply_layer(comparators, arr):
    """Apply a set of disjoint comparators to an array in-place (simultaneously)."""
    for i, j in comparators:
        if arr[i] > arr[j]:
            arr[i], arr[j] = arr[j], arr[i]

def _is_sorted(arr):
    for i in range(len(arr) - 1):
        if arr[i] > arr[i + 1]:
            return False
    return True

def _verify_network(layers):
    """Verify the network sorts all 2^N binary inputs using the zero-one principle."""
    n = N
    for mask in range(1 << n):
        arr = [(mask >> i) & 1 for i in range(n)]
        for layer in layers:
            _apply_layer(layer, arr)
        if not _is_sorted(arr):
            return False
    return True

def _count_comparators(layers):
    return sum(len(layer) for layer in layers)

def _get_depth(layers):
    return len(layers)

def _generate_seed_network():
    """Generate the odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        layers.append(layer)
    return layers

def _remove_comparator_from_layer(layers, layer_idx, comp_idx):
    """Remove a specific comparator from a specific layer."""
    new_layers = [list(layer) for layer in layers]
    new_layers[layer_idx].pop(comp_idx)
    return new_layers

def _add_comparator_to_layer(layers, layer_idx, i, j):
    """Add a comparator to a specific layer, ensuring disjointness."""
    new_layers = [list(layer) for layer in layers]
    # Check if i or j is already used in this layer
    for a, b in new_layers[layer_idx]:
        if i in (a, b) or j in (a, b):
            return None  # Cannot add due to conflict
    new_layers[layer_idx].append((i, j))
    return new_layers

def _local_search(layers, time_limit=50.0):
    """Local search: try removing comparators, then try adding them back if needed."""
    start_time = time.time()
    current_layers = [list(layer) for layer in layers]
    current_count = _count_comparators(current_layers)
    best_layers = [list(layer) for layer in current_layers]
    best_count = current_count
    
    # First, try to remove comparators
    while time.time() - start_time < time_limit:
        improved = False
        # Try removing each comparator
        for layer_idx in range(len(current_layers)):
            for comp_idx in range(len(current_layers[layer_idx])):
                if time.time() - start_time >= time_limit:
                    break
                # Try removing this comparator
                trial_layers = _remove_comparator_from_layer(current_layers, layer_idx, comp_idx)
                if trial_layers is None:
                    continue
                if _verify_network(trial_layers):
                    trial_count = _count_comparators(trial_layers)
                    if trial_count < current_count:
                        current_layers = trial_layers
                        current_count = trial_count
                        improved = True
                        if current_count < best_count:
                            best_layers = [list(layer) for layer in current_layers]
                            best_count = current_count
                        break
            if time.time() - start_time >= time_limit:
                break
            if improved:
                break
        if not improved:
            break
    
    # Now try to add comparators to improve depth or fix issues
    # This is less critical since we're minimizing count
    return best_layers

def _generate_random_layer():
    """Generate a random layer of disjoint comparators."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    i = 0
    while i < N - 1:
        a = wires[i]
        b = wires[i + 1]
        if a > b:
            a, b = b, a
        layer.append((a, b))
        i += 2
    return layer

def _mutate_network(layers, mutation_rate=0.3):
    """Mutate a network by modifying some layers."""
    new_layers = [list(layer) for layer in layers]
    for i in range(len(new_layers)):
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                # Replace layer with a new random layer
                new_layers[i] = _generate_random_layer()
            else:
                # Modify a single comparator in the layer
                if new_layers[i]:
                    idx = random.randint(0, len(new_layers[i]) - 1)
                    new_layers[i].pop(idx)
                    new_layers[i] = _generate_random_layer()
    return [tuple(layer) for layer in new_layers]

def _crossover_network(layers1, layers2):
    """Crossover two networks."""
    if len(layers1) != len(layers2):
        shorter, longer = (layers1, layers2) if len(layers1) < len(layers2) else (layers2, layers1)
        while len(shorter) < len(longer):
            shorter.append(_generate_random_layer())
        layers1, layers2 = (shorter, longer) if len(layers1) <= len(layers2) else (longer, shorter)
    
    new_layers = []
    for i in range(len(layers1)):
        if random.random() < 0.5:
            new_layers.append(list(layers1[i]))
        else:
            new_layers.append(list(layers2[i]))
    return [tuple(layer) for layer in new_layers]

def _optimize_network(initial_layers, time_limit=50.0):
    """Optimize the network using local search with comparator removal."""
    start_time = time.time()
    
    # Start with the seed network
    best_network = [list(layer) for layer in initial_layers]
    best_count = _count_comparators(best_network)
    
    # Verify the seed network
    if not _verify_network(best_network):
        return initial_layers
    
    # Apply local search
    optimized = _local_search(best_network, time_limit=time_limit)
    if _verify_network(optimized):
        opt_count = _count_comparators(optimized)
        if opt_count < best_count:
            best_network = optimized
            best_count = opt_count
    
    return best_network

def construct():
    # Generate the seed network
    seed_layers = _generate_seed_network()
    
    # Verify it's correct
    if not _verify_network(seed_layers):
        return [(i, j) for layer in seed_layers for i, j in layer]
    
    # Try to optimize
    try:
        optimized_layers = _optimize_network(seed_layers, time_limit=TIME_LIMIT)
        
        # Verify the optimized network
        if _verify_network(optimized_layers):
            comparators = []
            for layer in optimized_layers:
                for comp in layer:
                    comparators.append(comp)
            
            if len(comparators) <= MAX_COMPARATORS:
                return comparators
    except Exception:
        pass
    
    # Fallback to seed network
    comparators = []
    for layer in seed_layers:
        for comp in layer:
            comparators.append(comp)
    return comparators

# EVOLVE-BLOCK-END
