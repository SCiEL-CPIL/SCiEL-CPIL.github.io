"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096

def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    total = 1 << N
    # Generate all 2^N boolean inputs
    # Use a fast method: for each bit position, create the pattern
    inputs = np.empty((total, N), dtype=np.uint8)
    for i in range(N):
        # Bit i of x for x in 0..total-1
        inputs[:, i] = (np.arange(total, dtype=np.uint32) >> i) & 1
    
    for (i, j) in network:
        a = inputs[:, i].copy()
        b = inputs[:, j].copy()
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    
    # Check sorted: no 1 before 0 in any row
    # Equivalent: for each row, the sequence is non-decreasing
    for k in range(N - 1):
        if np.any(inputs[:, k] > inputs[:, k + 1]):
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
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
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
    
    # Phase 2: Local search with random order - try removing comparators in random order
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        # Use a random permutation approach to avoid bias
        while time.time() < deadline:
            if len(network) <= best_count:
                break
            # Generate random order of indices
            order = rng.permutation(len(network))
            found_removal = False
            for pos in order:
                if time.time() > deadline:
                    return best_network
                candidate = network[:pos] + network[pos+1:]
                if _verify(candidate):
                    network = candidate
                    best_network = list(network)
                    best_count = len(network)
                    found_removal = True
                    break
            if not found_removal:
                # No more single removals possible, break
                break
    
    # Phase 3: Layer-based local search - represent as layers and try mutations
    if time.time() < deadline:
        # Convert to layer representation
        def to_layers(net):
            """Convert flat network to list of layers (each layer is a set of non-conflicting comparators)."""
            layers = []
            used = set()
            i = 0
            while i < len(net):
                layer = []
                used_in_layer = set()
                while i < len(net):
                    comp = net[i]
                    if comp[0] not in used_in_layer and comp[1] not in used_in_layer:
                        layer.append(comp)
                        used_in_layer.add(comp[0])
                        used_in_layer.add(comp[1])
                        i += 1
                    else:
                        break
                layers.append(layer)
            return layers
        
        def from_layers(layers):
            """Convert layers back to flat network."""
            net = []
            for layer in layers:
                net.extend(layer)
            return net
        
        # Try to reduce depth by merging layers where possible
        layers = to_layers(network)
        
        # Try swapping comparators between adjacent layers to enable removals
        for _pass in range(3):
            if time.time() > deadline:
                break
            for l in range(len(layers) - 1):
                if time.time() > deadline:
                    break
                # Try moving a comparator from layer l to layer l+1 or vice versa
                for i in range(len(layers[l])):
                    if time.time() > deadline:
                        break
                    comp = layers[l][i]
                    # Try removing from layer l and adding to layer l+1
                    new_layer_l = layers[l][:i] + layers[l][i+1:]
                    # Check if comp can fit in layer l+1
                    can_fit = True
                    for c2 in layers[l+1]:
                        if comp[0] in c2 or comp[1] in c2:
                            can_fit = False
                            break
                    if can_fit:
                        new_layer_l1 = layers[l+1] + [comp]
                        # Reconstruct network
                        new_layers = layers[:l] + [new_layer_l] + [new_layer_l1] + layers[l+2:]
                        candidate = from_layers(new_layers)
                        if len(candidate) < len(network):
                            if _verify(candidate):
                                network = candidate
                                best_network = list(network)
                                best_count = len(network)
                                layers = to_layers(network)
                                break
                if time.time() > deadline:
                    break
    
    # Phase 4: Final greedy removal pass
    if time.time() < deadline:
        improved = True
        while improved:
            improved = False
            for idx in range(len(network) - 1, -1, -1):
                if time.time() > deadline:
                    break
                success, candidate = _try_remove(network, idx)
                if success:
                    network = candidate
                    improved = True
                    best_network = list(network)
                    best_count = len(network)
            if time.time() > deadline:
                break
    
    return best_network
# EVOLVE-BLOCK-END
