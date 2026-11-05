"""Improved sorting network search for 17 wires."""
import time
import numpy as np

N = 17
MAX_COMPARATORS = 4096
NUM_BOOLEANS = 1 << N  # 131072

def _verify(network):
    """Verify sorting network over all 2^N Boolean inputs using zero-one principle.
    
    Uses bit-packing for speed: each input is a 17-bit integer.
    """
    if not network:
        return N == 1
    
    # bits[x] will hold the current state of input x as a 17-bit integer
    # Initially bits[x] = x (the input itself)
    bits = np.arange(NUM_BOOLEANS, dtype=np.uint32).copy()
    
    for (i, j) in network:
        mask_i = 1 << i
        mask_j = 1 << j
        # Extract values of wire i and wire j
        bi = bits & mask_i
        bj = bits & mask_j
        # Compare-exchange: wire i gets min, wire j gets max
        # If bi < bj (numerically, since these are single bits), swap
        # bi < bj means bi=0, bj=1 -> after swap: bi=1, bj=0
        # bi >= bj means bi=1, bj=0 or equal -> no swap needed (or both same)
        # Actually: if bi=0 and bj=1: min=0 goes to i (already 0), max=1 goes to j (already 1) - no change
        # If bi=1 and bj=0: min=0 goes to i, max=1 goes to j - SWAP
        # So swap happens when bi=1 and bj=0, i.e., bi > bj
        swap_mask = (bi > bj)  # True where bi=1, bj=0
        # For swap: clear bit i, set bit j; for no swap: leave as is
        # New bits = bits XOR (mask_i | mask_j) where swap occurs
        xor_val = mask_i | mask_j
        bits = np.where(swap_mask, bits ^ xor_val, bits)
    
    # Check sorted: for each adjacent pair (k, k+1), we need bit_k <= bit_{k+1}
    # Violation: bit_k = 1 and bit_{k+1} = 0
    for k in range(N - 1):
        mask_k = 1 << k
        mask_k1 = 1 << (k + 1)
        # Check if there's any input where bit k = 1 and bit k+1 = 0
        violation = (bits & mask_k) != 0
        violation &= (bits & mask_k1) == 0
        if np.any(violation):
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

def _to_layers(net):
    """Convert flat network to list of layers (each layer is a set of non-conflicting comparators)."""
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
    if not _verify(network):
        return network
    
    best_network = list(network)
    best_count = len(network)
    
    # Phase 1: Greedy removal - repeatedly remove redundant comparators
    # Process from end to start to maintain stability
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
    
    # Phase 2: Random-order local search for removals
    if time.time() < deadline:
        rng = np.random.default_rng(42)
        while time.time() < deadline:
            if len(network) <= best_count:
                break
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
                break
    
    # Phase 3: Layer-based optimization - try to reduce depth and enable further removals
    if time.time() < deadline:
        layers = _to_layers(network)
        
        for _pass in range(5):
            if time.time() > deadline:
                break
            
            for l in range(len(layers) - 1):
                if time.time() > deadline:
                    break
                
                # Try moving comparators between adjacent layers
                for i in range(len(layers[l])):
                    if time.time() > deadline:
                        break
                    
                    comp = layers[l][i]
                    new_layer_l = layers[l][:i] + layers[l][i+1:]
                    
                    # Check if comp can fit in layer l+1
                    can_fit = True
                    for c2 in layers[l+1]:
                        if comp[0] in c2 or comp[1] in c2:
                            can_fit = False
                            break
                    
                    if can_fit:
                        new_layer_l1 = layers[l+1] + [comp]
                        new_layers = layers[:l] + [new_layer_l] + [new_layer_l1] + layers[l+2:]
                        candidate = _from_layers(new_layers)
                        if _verify(candidate):
                            network = candidate
                            best_network = list(network)
                            best_count = len(network)
                            layers = _to_layers(network)
                            break
                
                # Also try moving from l+1 to l
                for i in range(len(layers[l+1])):
                    if time.time() > deadline:
                        break
                    
                    comp = layers[l+1][i]
                    new_layer_l1 = layers[l+1][:i] + layers[l+1][i+1:]
                    
                    can_fit = True
                    for c2 in layers[l]:
                        if comp[0] in c2 or comp[1] in c2:
                            can_fit = False
                            break
                    
                    if can_fit:
                        new_layer_l = layers[l] + [comp]
                        new_layers = layers[:l] + [new_layer_l] + [new_layer_l1] + layers[l+2:]
                        candidate = _from_layers(new_layers)
                        if _verify(candidate):
                            network = candidate
                            best_network = list(network)
                            best_count = len(network)
                            layers = _to_layers(network)
                            break
    
    # Phase 4: Insert-remove strategy - add a comparator, then try removing two
    if time.time() < deadline:
        # Only try a subset of insertions to save time
        rng = np.random.default_rng(99)
        attempts = 0
        max_attempts = 200
        
        while time.time() < deadline and attempts < max_attempts:
            attempts += 1
            
            # Pick a random insertion position and random comparator
            ins_pos = rng.integers(0, len(network) + 1)
            i, j = rng.choice(N, 2, replace=False)
            if i > j:
                i, j = j, i
            new_comp = (i, j)
            
            candidate = network[:ins_pos] + [new_comp] + network[ins_pos:]
            
            # Now try to remove two comparators from this larger network
            removed = 0
            improved_net = list(candidate)
            
            # Try removing from the end first
            for _ in range(2):
                if time.time() > deadline:
                    break
                found = False
                for idx in range(len(improved_net) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    cand2 = improved_net[:idx] + improved_net[idx+1:]
                    if _verify(cand2):
                        improved_net = cand2
                        removed += 1
                        found = True
                        break
                if not found:
                    break
            
            if removed >= 1 and len(improved_net) < len(network):
                network = improved_net
                best_network = list(network)
                best_count = len(network)
    
    # Phase 5: Final intensive local search with multiple random orders
    if time.time() < deadline:
        rng = np.random.default_rng(123)
        for _round in range(20):
            if time.time() > deadline:
                break
            if len(network) <= best_count:
                break
            order = rng.permutation(len(network))
            found = False
            for pos in order:
                if time.time() > deadline:
                    break
                candidate = network[:pos] + network[pos+1:]
                if _verify(candidate):
                    network = candidate
                    best_network = list(network)
                    best_count = len(network)
                    found = True
                    break
            if not found:
                break
    
    # Phase 6: Final greedy pass
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
