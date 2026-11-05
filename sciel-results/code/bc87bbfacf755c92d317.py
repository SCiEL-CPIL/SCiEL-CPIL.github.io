"""Improved sorting network search for 17 wires."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096

# Precompute the full set of 2^N boolean inputs as a 2D uint8 array
# Shape: (2^N, N) where each row is one boolean input
_TOTAL = 1 << N
_ALL_INPUTS = np.empty((_TOTAL, N), dtype=np.uint8)
for _i in range(N):
    _ALL_INPUTS[:, _i] = (np.arange(_TOTAL, dtype=np.uint32) >> _i) & 1

def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle."""
    # Work on a copy of the inputs
    inputs = _ALL_INPUTS.copy()
    
    for (i, j) in network:
        a = inputs[:, i]
        b = inputs[:, j]
        # min goes to i, max goes to j
        # We need to be careful: a and b are views, so we need temp storage
        tmp = np.minimum(a, b)
        inputs[:, i] = tmp
        inputs[:, j] = np.maximum(a, b)
    
    # Check sorted: each row should be non-increasing
    # A row is sorted if for all k, inputs[k] >= inputs[k+1]
    # Unsorted if there exists k where inputs[k] < inputs[k+1]
    diffs = np.diff(inputs, axis=1)  # shape (TOTAL, N-1)
    # If any diff is positive, that row is unsorted
    if np.any(diffs > 0):
        return False
    return True

def _verify_fast(network):
    """Faster verification using in-place operations on a single buffer."""
    inputs = _ALL_INPUTS.copy()
    
    for (i, j) in network:
        a = inputs[:, i]
        b = inputs[:, j]
        # Use a temporary to avoid overwriting before we're done
        # min(a,b) -> i, max(a,b) -> j
        # We can do this with two temp arrays or use the fact that 
        # min+max = a+b, so max = a+b-min
        mn = np.minimum(a, b)
        mx = a + b - mn  # since a,b are 0 or 1
        inputs[:, i] = mn
        inputs[:, j] = mx
    
    # Check if sorted
    for k in range(N - 1):
        if np.any(inputs[:, k] < inputs[:, k + 1]):
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
    if _verify_fast(candidate):
        return True, candidate
    return False, None

def _count_comparators(network):
    """Count total comparators."""
    return len(network)

def _compute_depth(network):
    """Compute parallel depth of the network."""
    if not network:
        return 0
    wire_free = [0] * N
    depth = 0
    for (i, j) in network:
        d = max(wire_free[i], wire_free[j]) + 1
        depth = max(depth, d)
        wire_free[i] = d
        wire_free[j] = d
    return depth

def _layered_network(network):
    """Convert flat network to list of layers (each layer is a set of non-conflicting comparators)."""
    layers = []
    i = 0
    while i < len(network):
        layer = []
        used_in_layer = set()
        while i < len(network):
            comp = network[i]
            if comp[0] not in used_in_layer and comp[1] not in used_in_layer:
                layer.append(comp)
                used_in_layer.add(comp[0])
                used_in_layer.add(comp[1])
                i += 1
            else:
                break
        layers.append(layer)
    return layers

def _from_layers(layers):
    """Convert layers back to flat network."""
    net = []
    for layer in layers:
        net.extend(layer)
    return net

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    # Start with odd-even transposition network
    network = _odd_even_transposition(N)
    
    # Verify it's correct
    if not _verify_fast(network):
        return network
    
    best_network = list(network)
    best_count = len(network)
    
    # Phase 1: Greedy removal - repeatedly remove redundant comparators from the end
    # Comparators later in the network are more likely to be redundant
    improved = True
    while improved:
        improved = False
        # Try from the end first, then work backwards
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return best_network
            
            success, candidate = _try_remove(network, idx)
            if success:
                network = candidate
                improved = True
                best_network = list(network)
                best_count = len(network)
    
    # Phase 2: Multi-pass greedy with different orderings
    rng = np.random.default_rng(42)
    for _pass in range(5):
        if time.time() > deadline:
            break
        
        # Try removals in a random order, but bias towards later positions
        n_comps = len(network)
        # Create a weighted random order: later indices get higher priority
        weights = np.linspace(1, n_comps, n_comps)
        order = np.argsort(rng.random(n_comps) / weights)  # Higher weight = more likely to be tried early
        
        found_any = False
        for pos in order:
            if time.time() > deadline:
                return best_network
            
            candidate = network[:pos] + network[pos+1:]
            if _verify_fast(candidate):
                network = candidate
                best_network = list(network)
                best_count = len(network)
                found_any = True
                # Restart with new network
                break
        
        if not found_any:
            break
    
    # Phase 3: Try removing pairs of comparators (remove one, then try to remove another)
    if time.time() < deadline:
        for _pass in range(3):
            if time.time() > deadline:
                break
            
            n_comps = len(network)
            # Try each position as the first removal
            for idx1 in range(n_comps - 1, -1, -1):
                if time.time() > deadline:
                    return best_network
                
                # Try removing idx1
                cand1 = network[:idx1] + network[idx1+1:]
                if not _verify_fast(cand1):
                    continue
                
                # Now try removing from cand1
                found_pair = False
                for idx2 in range(len(cand1) - 1, -1, -1):
                    if time.time() > deadline:
                        return best_network
                    
                    cand2 = cand1[:idx2] + cand1[idx2+1:]
                    if _verify_fast(cand2):
                        network = cand2
                        best_network = list(network)
                        best_count = len(network)
                        found_pair = True
                        break
                
                if found_pair:
                    break
            
            # If we found a pair removal, loop again
            # Otherwise, no more pair removals possible in this pass
    
    # Phase 4: Layer-based reorganization to enable further removals
    if time.time() < deadline:
        def to_layers(net):
            layers = []
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
            net = []
            for layer in layers:
                net.extend(layer)
            return net
        
        # Try swapping comparators between adjacent layers to create new removal opportunities
        for _pass in range(5):
            if time.time() > deadline:
                break
            
            layers = to_layers(network)
            found_swap = False
            
            for l in range(len(layers) - 1):
                if time.time() > deadline:
                    return best_network
                
                for i in range(len(layers[l])):
                    if time.time() > deadline:
                        return best_network
                    
                    comp = layers[l][i]
                    # Try moving comp from layer l to layer l+1
                    can_fit = True
                    for c2 in layers[l+1]:
                        if comp[0] in c2 or comp[1] in c2:
                            can_fit = False
                            break
                    
                    if not can_fit:
                        continue
                    
                    new_layers = [list(x) for x in layers]
                    new_layers[l] = new_layers[l][:i] + new_layers[l][i+1:]
                    new_layers[l+1] = new_layers[l+1] + [comp]
                    
                    candidate = from_layers(new_layers)
                    if _verify_fast(candidate):
                        network = candidate
                        best_network = list(network)
                        best_count = len(network)
                        found_swap = True
                        break
                
                if found_swap:
                    break
            
            if not found_swap:
                break
        
        # After reorganization, do another greedy removal pass
        if time.time() < deadline:
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
    
    # Phase 5: Final intensive local search with random restarts
    if time.time() < deadline:
        rng = np.random.default_rng(123)
        for _round in range(20):
            if time.time() > deadline:
                break
            
            order = rng.permutation(len(network))
            found = False
            for pos in order:
                if time.time() > deadline:
                    break
                candidate = network[:pos] + network[pos+1:]
                if _verify_fast(candidate):
                    network = candidate
                    best_network = list(network)
                    best_count = len(network)
                    found = True
                    break
            if not found:
                break
    
    return best_network
# EVOLVE-BLOCK-END
