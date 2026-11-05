"""Size-efficient sorting networks on 17 wires – search-based construction."""

import time
import random

N = 17

# Deadline guard: kill search well before 90s wall-clock limit
_START_TIME = time.time()
_DEADLINE = 78.0  # seconds; leaves 12s safety margin


def _now():
    return time.time()


def _time_exceeded():
    return (_now() - _START_TIME) >= _DEADLINE


def _apply_comparator(wires, i, j):
    """Apply a compare-exchange: min goes to wire i, max to wire j."""
    if wires[i] > wires[j]:
        wires[i], wires[j] = wires[j], wires[i]
    return wires


def _is_sorted(wires):
    return all(wires[i] <= wires[i + 1] for i in range(len(wires) - 1))


def _verify_booleans_v2(network):
    """
    Vectorized verification using Python big integers.
    Each wire is a big int; bit m represents input m's value on that wire.
    Comparator (i,j): new_i = i & j, new_j = i | j (for 0/1 values).
    """
    total = 1 << N
    mask_all = (1 << total) - 1
    
    # Initialize
    wires = [0] * N
    for k in range(N):
        # Bit m of wire k = (m >> k) & 1
        block = 1 << k
        val = 0
        for m in range(block, total, 2 * block):
            bits_to_set = (1 << block) - 1
            val |= bits_to_set << m
        wires[k] = val
    
    # Apply comparators
    for (i, j) in network:
        a = wires[i]
        b = wires[j]
        wires[i] = a & b
        wires[j] = a | b
    
    # Check sorted
    for k in range(N - 1):
        if wires[k] & ~wires[k + 1] & mask_all:
            return False
    return True


def _verify_sample(network, samples):
    """Verify network on a set of sample inputs (tuples of 0/1)."""
    for inp in samples:
        wires = list(inp)
        for (i, j) in network:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        if not _is_sorted(wires):
            return False
    return True


def _odd_even_transposition(n=N):
    """Standard odd-even transposition sorting network."""
    comp = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comp.append((i, i + 1))
    return comp


def _depth_of_network(network, n=N):
    """Compute the depth (number of layers) of a network."""
    if not network:
        return 0
    last_use = [-1] * n
    depth = [0] * len(network)
    for idx in range(len(network)):
        i, j = network[idx]
        d = 0
        if last_use[i] >= 0:
            d = max(d, depth[last_use[i]] + 1)
        if last_use[j] >= 0:
            d = max(d, depth[last_use[j]] + 1)
        depth[idx] = d
        last_use[i] = idx
        last_use[j] = idx
    return max(depth) + 1 if network else 0


def _generate_random_layer(n=N, rng=None):
    """Generate a random valid layer (set of non-crossing comparators)."""
    if rng is None:
        rng = random
    used = [False] * n
    layer = []
    wires = list(range(n))
    rng.shuffle(wires)
    i = 0
    while i < n - 1:
        a = wires[i]
        b = wires[i + 1]
        if a > b:
            a, b = b, a
        if not used[a] and not used[b]:
            layer.append((a, b))
            used[a] = True
            used[b] = True
            i += 2
        else:
            i += 1
    return layer


def _layers_to_network(layers):
    """Convert list of layers (each a list of (i,j) pairs) to flat network."""
    network = []
    for layer in layers:
        network.extend(layer)
    return network


def _network_to_layers(network, n=N):
    """Convert flat network to list of layers by greedy layer assignment."""
    if not network:
        return []
    last_use = [-1] * n
    layers = []
    current_layer = []
    current_depth = 0
    for (i, j) in network:
        # Check if this comparator can be in current layer
        if (last_use[i] == current_depth - 1) or (last_use[j] == current_depth - 1):
            # Must start a new layer
            if current_layer:
                layers.append(current_layer)
            current_layer = [(i, j)]
            current_depth += 1
        else:
            current_layer.append((i, j))
        last_use[i] = current_depth - 1
        last_use[j] = current_depth - 1
    if current_layer:
        layers.append(current_layer)
    return layers


def _mutate_layer(layer, n=N, rng=None):
    """Mutate a single layer by swapping, adding, or removing comparators."""
    if rng is None:
        rng = random
    if not layer:
        return _generate_random_layer(n, rng)
    
    # Convert to set for quick lookup
    comp_set = set(layer)
    used = [False] * n
    for (a, b) in layer:
        used[a] = True
        used[b] = True
    
    mutation_type = rng.random()
    
    if mutation_type < 0.3:
        # Remove a random comparator
        idx = rng.randint(0, len(layer) - 1)
        a, b = layer[idx]
        del layer[idx]
        used[a] = False
        used[b] = False
    elif mutation_type < 0.6:
        # Add a random comparator
        # Find available wires
        available = [i for i in range(n) if not used[i]]
        if len(available) >= 2:
            i, j = rng.sample(available, 2)
            if i > j:
                i, j = j, i
            layer.append((i, j))
            used[i] = True
            used[j] = True
    else:
        # Swap two comparators' positions or replace one
        if len(layer) >= 2:
            idx1, idx2 = rng.sample(range(len(layer)), 2)
            # Swap the pairs
            layer[idx1], layer[idx2] = layer[idx2], layer[idx1]
        elif len(layer) == 1:
            # Replace the single comparator
            a, b = layer[0]
            used[a] = False
            used[b] = False
            available = [i for i in range(n) if not used[i]]
            if len(available) >= 2:
                i, j = rng.sample(available, 2)
                if i > j:
                    i, j = j, i
                layer[0] = (i, j)
                used[i] = True
                used[j] = True
    
    return layer


def _genetic_search(rng, max_time=60.0, pop_size=50, elite_size=5):
    """
    Genetic algorithm that represents the sorting network as a sequence of parallel layers.
    Each layer is a matching of comparators. Apply mutations that modify layers.
    """
    start_time = _now()
    max_time = min(max_time, _DEADLINE - (_now() - _START_TIME) - 2.0)
    
    # Initialize population
    population = []
    # Start with odd-even transposition
    base_network = _odd_even_transposition(N)
    base_layers = _network_to_layers(base_network, N)
    population.append(base_layers)
    
    # Add random networks
    for _ in range(pop_size - 1):
        num_layers = rng.randint(8, 16)
        layers = []
        for _ in range(num_layers):
            layers.append(_generate_random_layer(N, rng))
        population.append(layers)
    
    # Evaluate initial population
    def evaluate(layers):
        network = _layers_to_network(layers)
        if not network:
            return 10000
        # Quick sample check
        samples = [tuple([0]*N), tuple([1]*N)]
        for _ in range(20):
            samples.append(tuple(rng.randint(0, 1) for _ in range(N)))
        if not _verify_sample(network, samples):
            return 10000
        return len(network)
    
    scores = [evaluate(layers) for layers in population]
    
    best_layers = None
    best_score = min(scores)
    if best_score < 10000:
        best_layers = population[scores.index(best_score)]
    
    # Genetic loop
    gen = 0
    while not _time_exceeded() and (_now() - start_time) < max_time:
        gen += 1
        
        # Sort by fitness
        sorted_pairs = sorted(zip(scores, population), key=lambda x: x[0])
        elite = [layers for _, layers in sorted_pairs[:elite_size]]
        
        # Create new population
        new_population = list(elite)
        
        while len(new_population) < pop_size:
            # Select parents
            parent1 = sorted_pairs[rng.randint(0, elite_size - 1)][1]
            parent2 = sorted_pairs[rng.randint(0, elite_size - 1)][1]
            
            # Crossover: take layers from parents
            child_layers = []
            for i in range(max(len(parent1), len(parent2))):
                if i < len(parent1) and i < len(parent2):
                    if rng.random() < 0.5:
                        child_layers.append(list(parent1[i]))
                    else:
                        child_layers.append(list(parent2[i]))
                elif i < len(parent1):
                    child_layers.append(list(parent1[i]))
                else:
                    child_layers.append(list(parent2[i]))
            
            # Mutation
            for i in range(len(child_layers)):
                if rng.random() < 0.3:
                    child_layers[i] = _mutate_layer(child_layers[i], N, rng)
            
            # Occasionally add or remove a layer
            if rng.random() < 0.1 and len(child_layers) > 5:
                del child_layers[rng.randint(0, len(child_layers) - 1)]
            elif rng.random() < 0.1:
                child_layers.append(_generate_random_layer(N, rng))
            
            new_population.append(child_layers)
        
        population = new_population
        scores = [evaluate(layers) for layers in population]
        
        # Update best
        current_best = min(scores)
        if current_best < best_score and current_best < 10000:
            best_score = current_best
            best_layers = population[scores.index(best_score)]
        
        # Check time
        if _time_exceeded():
            break
    
    if best_layers is None:
        return _odd_even_transposition(N)
    
    return _layers_to_network(best_layers)


def construct():
    """
    Construct a sorting network on N=17 wires.
    Uses search: try known constructions, then genetic algorithm within time budget.
    Returns ordered list of (i, j) pairs.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Start with odd-even transposition (136 comparators for 17 wires)
    base_network = _odd_even_transposition(N)
    best = base_network
    best_len = len(best)
    
    rng = random.Random(42)
    
    # Generate sample inputs for quick verification
    samples = []
    samples.append(tuple([0] * N))
    samples.append(tuple([1] * N))
    for _ in range(200):
        samples.append(tuple(rng.randint(0, 1) for _ in range(N)))
    
    # Verify base network
    if not _verify_booleans_v2(best):
        return best
    
    # Phase 1: Greedy removal of redundant comparators
    improved = True
    while improved and not _time_exceeded():
        improved = False
        for idx in range(len(best)):
            if _time_exceeded():
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_booleans_v2(candidate):
                best = candidate
                improved = True
                break
    
    best_len = len(best)
    
    # Phase 2: Genetic algorithm
    if not _time_exceeded():
        ga_network = _genetic_search(rng, max_time=60.0)
        if len(ga_network) < best_len:
            # Verify the GA result
            if _verify_booleans_v2(ga_network):
                best = ga_network
                best_len = len(best)
    
    # Phase 3: Try greedy removal on the best found so far
    improved = True
    while improved and not _time_exceeded():
        improved = False
        for idx in range(len(best)):
            if _time_exceeded():
                break
            candidate = best[:idx] + best[idx+1:]
            if _verify_booleans_v2(candidate):
                best = candidate
                improved = True
                break
    
    return best
