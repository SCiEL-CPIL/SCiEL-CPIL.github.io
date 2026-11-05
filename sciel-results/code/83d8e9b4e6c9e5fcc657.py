"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096

def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    total = 1 << N
    # Precompute all 2^N boolean inputs once
    # Use bit patterns: input x has bit i set if (x >> i) & 1
    inputs = np.empty((total, N), dtype=np.uint8)
    for i in range(N):
        inputs[:, i] = (np.arange(total, dtype=np.uint32) >> i) & 1
    
    for (i, j) in network:
        a = inputs[:, i].copy()
        b = inputs[:, j].copy()
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    
    # Check sorted: no 1 before 0 in any row
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
            return False
    return True

def _network_from_layers(layers):
    """Flatten a list of layers (each a list of (i,j) pairs) into a flat list."""
    comps = []
    for layer in layers:
        for c in layer:
            comps.append(c)
    return comps

def _layers_from_network(network):
    """Group a flat list of comparators into layers (maximally parallel)."""
    layers = []
    used = set()
    for c in network:
        placed = False
        for li, layer in enumerate(layers):
            wires_in_layer = set()
            for (a, b) in layer:
                wires_in_layer.add(a)
                wires_in_layer.add(b)
            if c[0] not in wires_in_layer and c[1] not in wires_in_layer:
                layer.append(c)
                placed = True
                break
        if not placed:
            layers.append([c])
    return layers

def _sample_verify(network, n_samples, rng):
    """Quick check using random Boolean inputs (necessary but not sufficient)."""
    for _ in range(n_samples):
        x = rng.integers(0, 2, size=N)
        for (i, j) in network:
            a, b = x[i], x[j]
            if a > b:
                x[i], x[j] = b, a
        for k in range(N - 1):
            if x[k] > x[k+1]:
                return False
    return True

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    comps = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comps.append((i, i + 1))
    return comps

def _try_remove(network, idx):
    """Try removing comparator at index idx, return (success, new_network)."""
    candidate = network[:idx] + network[idx+1:]
    if _verify(candidate):
        return True, candidate
    return False, None

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin before 90s kill
    rng = np.random.default_rng(42)
    
    # Start with odd-even transposition network
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    if not _verify(network):
        return network
    
    best_network = list(network)
    best_count = len(network)
    
    # Phase 1: Greedy removal - repeatedly remove redundant comparators
    improved = True
    while improved:
        improved = False
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return best_network
            
            success, candidate = _try_remove(network, idx)
            if success:
                network = candidate
                improved = True
                best_network = list(network)
                best_count = len(network)
    
    # Phase 2: Layer-based genetic local search
    # Represent the network as layers of parallel comparators
    layers = _layers_from_network(network)
    
    def _layer_count_comparators(layers):
        return sum(len(l) for l in layers)
    
    def _mutate_layers(layers, rng):
        """Apply a random mutation to the layer representation."""
        new_layers = [list(l) for l in layers]
        mutation_type = rng.integers(0, 5)
        
        if mutation_type == 0:
            # Remove a random comparator from a random layer
            if len(new_layers) > 0:
                li = rng.integers(0, len(new_layers))
                if len(new_layers[li]) > 0:
                    ci = rng.integers(0, len(new_layers[li]))
                    new_layers[li].pop(ci)
                    # Remove empty layers
                    new_layers = [l for l in new_layers if len(l) > 0]
        
        elif mutation_type == 1:
            # Move a comparator to a different layer
            if len(new_layers) > 1:
                li = rng.integers(0, len(new_layers))
                if len(new_layers[li]) > 0:
                    ci = rng.integers(0, len(new_layers[li]))
                    comp = new_layers[li].pop(ci)
                    # Find a layer where this comparator can be placed
                    target = None
                    for ti, tl in enumerate(new_layers):
                        if ti == li:
                            continue
                        wires = set()
                        for (a, b) in tl:
                            wires.add(a)
                            wires.add(b)
                        if comp[0] not in wires and comp[1] not in wires:
                            target = ti
                            break
                    if target is not None:
                        new_layers[target].append(comp)
                    else:
                        new_layers[li].append(comp)
                    new_layers = [l for l in new_layers if len(l) > 0]
        
        elif mutation_type == 2:
            # Swap two comparators within the same layer
            for li in range(len(new_layers)):
                if len(new_layers[li]) >= 2:
                    ci1 = rng.integers(0, len(new_layers[li]))
                    ci2 = rng.integers(0, len(new_layers[li]))
                    new_layers[li][ci1], new_layers[li][ci2] = new_layers[li][ci2], new_layers[li][ci1]
                    break
        
        elif mutation_type == 3:
            # Swap comparators between two layers
            if len(new_layers) >= 2:
                li1 = rng.integers(0, len(new_layers))
                li2 = rng.integers(0, len(new_layers))
                if li1 != li2 and len(new_layers[li1]) > 0 and len(new_layers[li2]) > 0:
                    ci1 = rng.integers(0, len(new_layers[li1]))
                    ci2 = rng.integers(0, len(new_layers[li2]))
                    c1 = new_layers[li1][ci1]
                    c2 = new_layers[li2][ci2]
                    # Check if swap preserves parallelism
                    wires1 = set()
                    for (a, b) in new_layers[li1]:
                        if (a, b) != c1:
                            wires1.add(a)
                            wires1.add(b)
                    wires2 = set()
                    for (a, b) in new_layers[li2]:
                        if (a, b) != c2:
                            wires2.add(a)
                            wires2.add(b)
                    if c2[0] not in wires1 and c2[1] not in wires1 and c1[0] not in wires2 and c1[1] not in wires2:
                        new_layers[li1][ci1] = c2
                        new_layers[li2][ci2] = c1
        
        elif mutation_type == 4:
            # Add a random comparator to a random layer
            li = rng.integers(0, len(new_layers)) if new_layers else 0
            if not new_layers:
                new_layers.append([])
            i = rng.integers(0, N)
            j = rng.integers(i + 1, N)
            comp = (i, j)
            wires = set()
            for (a, b) in new_layers[li]:
                wires.add(a)
                wires.add(b)
            if comp[0] not in wires and comp[1] not in wires:
                new_layers[li].append(comp)
        
        return new_layers
    
    def _greedy_remove_from_layers(layers, rng):
        """Try to remove redundant comparators from the layer representation."""
        improved = True
        while improved:
            improved = False
            for li in range(len(layers) - 1, -1, -1):
                for ci in range(len(layers[li]) - 1, -1, -1):
                    # Try removing this comparator
                    new_layers = [list(l) for l in layers]
                    new_layers[li].pop(ci)
                    new_layers = [l for l in new_layers if len(l) > 0]
                    flat = _network_from_layers(new_layers)
                    if _verify(flat):
                        layers = new_layers
                        improved = True
                        break
                if improved:
                    break
        return layers
    
    # Genetic local search loop
    pop_size = 8
    current_layers = [list(l) for l in layers]
    current_count = _layer_count_comparators(current_layers)
    
    generations = 0
    while time.time() < deadline - 5.0:
        generations += 1
        # Try to improve current solution
        current_layers = _greedy_remove_from_layers(current_layers, rng)
        current_count = _layer_count_comparators(current_layers)
        
        if current_count < best_count:
            best_network = _network_from_layers(current_layers)
            best_count = current_count
            network = list(best_network)
            layers = _layers_from_network(network)
        
        # Generate offspring
        for _ in range(pop_size):
            child_layers = _mutate_layers(current_layers, rng)
            child_flat = _network_from_layers(child_layers)
            
            # Quick sample check first
            if not _sample_verify(child_flat, 50, rng):
                continue
            
            # Full verification
            if _verify(child_flat):
                child_count = len(child_flat)
                # Greedy remove from child
                child_layers2 = _greedy_remove_from_layers(child_layers, rng)
                child_flat2 = _network_from_layers(child_layers2)
                child_count2 = len(child_flat2)
                
                if child_count2 < best_count:
                    best_network = list(child_flat2)
                    best_count = child_count2
                    network = list(best_network)
                    layers = _layers_from_network(network)
                    current_layers = child_layers2
                    current_count = child_count2
        
        # Occasionally re-seed from best
        if generations % 5 == 0:
            current_layers = _layers_from_network(best_network)
            current_count = _layer_count_comparators(current_layers)
    
    # Final polish: greedy removal on best
    final_network = list(best_network)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(final_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = final_network[:idx] + final_network[idx+1:]
            if _verify(candidate):
                final_network = candidate
                improved = True
                if len(final_network) < best_count:
                    best_network = list(final_network)
                    best_count = len(final_network)
    
    return best_network
# EVOLVE-BLOCK-END
