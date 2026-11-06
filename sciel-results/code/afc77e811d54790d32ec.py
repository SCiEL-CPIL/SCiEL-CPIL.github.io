"""Size-efficient sorting networks on 17 wires: layer-evolution approach."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs (2^17 = 131072)
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_inputs(n_samples=4096):
    all_inputs = _get_all_inputs()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
    return all_inputs[idx]

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_fast(comparators, inputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs = _get_sample_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def odd_even_merge_sort_network(n):
    """Generate an odd-even merge sort network for n wires."""
    if n == 1:
        return []
    
    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        half = n // 2
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        merged = _odd_even_merge(n)
        return left + right + merged
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        comparators = []
        if n % 2 == 0:
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        return comparators
    
    return _build(n)

def odd_even_transposition_network(n):
    """Classic odd-even transposition sort network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def build_layers(comparators):
    """Build depth-parallel layers from a sequential comparator list."""
    layers = []
    current_layer = []
    used_wires = set()
    for (i, j) in comparators:
        if i in used_wires or j in used_wires:
            layers.append(current_layer)
            current_layer = []
            used_wires = set()
        current_layer.append((i, j))
        used_wires.add(i)
        used_wires.add(j)
    if current_layer:
        layers.append(current_layer)
    return layers

def flatten_layers(layers):
    """Flatten layers back to a sequential list."""
    result = []
    for layer in layers:
        result.extend(layer)
    return result

def _random_layer(max_comparators=None):
    """Generate a random layer of disjoint comparators."""
    wires = list(range(N))
    random.shuffle(wires)
    comparators = []
    for k in range(0, len(wires) - 1, 2):
        a, b = wires[k], wires[k + 1]
        if a > b:
            a, b = b, a
        comparators.append((a, b))
        if max_comparators and len(comparators) >= max_comparators:
            break
    return comparators

def _layer_is_valid(layer):
    """Check that a layer has no wire conflicts."""
    used = set()
    for (i, j) in layer:
        if i in used or j in used:
            return False
        used.add(i)
        used.add(j)
    return True

def _evolve_layer(layer, n_wires):
    """Mutate a single layer: add, remove, or swap comparators."""
    new_layer = list(layer)
    op = random.random()
    
    if op < 0.3 and len(new_layer) > 0:
        # Remove a random comparator
        idx = random.randint(0, len(new_layer) - 1)
        new_layer.pop(idx)
    elif op < 0.6:
        # Add a random comparator on unused wires
        used = set()
        for (i, j) in new_layer:
            used.add(i)
            used.add(j)
        available = [w for w in range(n_wires) if w not in used]
        if len(available) >= 2:
            a, b = random.sample(available, 2)
            if a > b:
                a, b = b, a
            new_layer.append((a, b))
    else:
        # Swap a comparator: pick one and reassign to different wires
        if len(new_layer) > 0:
            idx = random.randint(0, len(new_layer) - 1)
            used = set()
            for k, (i, j) in enumerate(new_layer):
                if k != idx:
                    used.add(i)
                    used.add(j)
            available = [w for w in range(n_wires) if w not in used]
            if len(available) >= 2:
                a, b = random.sample(available, 2)
                if a > b:
                    a, b = b, a
                new_layer[idx] = (a, b)
    
    return new_layer

def _evolve_network(layers, mutation_rate=0.3):
    """Evolve a network represented as layers."""
    new_layers = []
    for layer in layers:
        if random.random() < mutation_rate:
            new_layer = _evolve_layer(layer, N)
        else:
            new_layer = list(layer)
        new_layers.append(new_layer)
    
    # Occasionally add or remove a layer
    r = random.random()
    if r < 0.1 and len(new_layers) < 30:
        # Add a layer
        pos = random.randint(0, len(new_layers))
        new_layers.insert(pos, _random_layer())
    elif r < 0.2 and len(new_layers) > 3:
        # Remove a layer
        pos = random.randint(0, len(new_layers) - 1)
        new_layers.pop(pos)
    
    return new_layers

def _network_from_layers(layers):
    """Convert layers to flat comparator list."""
    return flatten_layers(layers)

def _count_comparators(layers):
    """Count total comparators in a layer representation."""
    return sum(len(layer) for layer in layers)

def _is_sorted_from_layers(layers):
    """Check if network (as layers) sorts correctly using full verification."""
    comparators = flatten_layers(layers)
    return is_sorted_network(comparators)

def _is_sorted_fast_from_layers(layers, sample=None):
    """Quick check using sample inputs."""
    comparators = flatten_layers(layers)
    return is_sorted_fast(comparators, sample)

def _local_search_layers(layers, sample, deadline, max_rounds=2000):
    """
    Local search on layer representation.
    Tries to remove comparators from individual layers.
    """
    best_layers = [list(layer) for layer in layers]
    best_count = _count_comparators(best_layers)
    no_improve_count = 0
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Pick a random layer and try removing a comparator from it
        if len(best_layers) == 0:
            break
        
        layer_idx = random.randint(0, len(best_layers) - 1)
        if len(best_layers[layer_idx]) == 0:
            continue
        
        comp_idx = random.randint(0, len(best_layers[layer_idx]) - 1)
        
        # Create candidate with this comparator removed
        candidate_layers = [list(layer) for layer in best_layers]
        candidate_layers[layer_idx] = list(best_layers[layer_idx])
        candidate_layers[layer_idx].pop(comp_idx)
        
        # Remove empty layers
        candidate_layers = [l for l in candidate_layers if len(l) > 0]
        
        cand_count = _count_comparators(candidate_layers)
        
        # Quick check
        if _is_sorted_fast_from_layers(candidate_layers, sample):
            # Full check
            if _is_sorted_from_layers(candidate_layers):
                best_layers = candidate_layers
                best_count = cand_count
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        if no_improve_count > 200:
            # Try a mutation to escape local minimum
            best_layers = _evolve_network(best_layers, mutation_rate=0.5)
            # Verify mutated network
            if not _is_sorted_network(flatten_layers(best_layers)):
                # Revert - keep original best
                best_layers = [list(layer) for layer in best_layers]
            no_improve_count = 0
    
    return best_layers, best_count

def _greedy_layer_removal(layers, sample, deadline):
    """
    Systematic greedy removal: try removing each comparator from each layer.
    """
    best_layers = [list(layer) for layer in layers]
    best_count = _count_comparators(best_layers)
    
    while time.time() < deadline:
        removed_any = False
        for li in range(len(best_layers)):
            if time.time() > deadline:
                break
            for ci in range(len(best_layers[li])):
                if time.time() > deadline:
                    break
                candidate_layers = [list(layer) for layer in best_layers]
                candidate_layers[li] = list(best_layers[li])
                candidate_layers[li].pop(ci)
                candidate_layers = [l for l in candidate_layers if len(l) > 0]
                
                if _is_sorted_fast_from_layers(candidate_layers, sample):
                    if _is_sorted_from_layers(candidate_layers):
                        best_layers = candidate_layers
                        best_count = _count_comparators(best_layers)
                        removed_any = True
                        break
            if removed_any:
                break
        
        if not removed_any:
            break
    
    return best_layers, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Start with odd-even merge sort network
    base_comparators = odd_even_merge_sort_network(N)
    
    # Verify base
    if not is_sorted_network(base_comparators):
        base_comparators = odd_even_transposition_network(N)
    
    # Convert to layers
    layers = build_layers(base_comparators)
    best_count = len(base_comparators)
    
    # Get sample for quick checks
    sample = _get_sample_inputs(4096)
    
    # Phase 1: Greedy layer removal
    if time.time() < deadline - 20:
        layers, best_count = _greedy_layer_removal(layers, sample, deadline - 20)
    
    # Phase 2: Local search on layers
    if time.time() < deadline - 10:
        layers, best_count = _local_search_layers(layers, sample, deadline - 10, max_rounds=3000)
    
    # Phase 3: Another greedy pass
    if time.time() < deadline - 5:
        layers, best_count = _greedy_layer_removal(layers, sample, deadline - 5)
    
    # Phase 4: More local search with higher mutation
    if time.time() < deadline - 3:
        layers, best_count = _local_search_layers(layers, sample, deadline - 3, max_rounds=2000)
    
    # Final verification
    final_comparators = flatten_layers(layers)
    if not is_sorted_network(final_comparators):
        # Fallback to known-good
        final_comparators = odd_even_transposition_network(N)
    
    return final_comparators
