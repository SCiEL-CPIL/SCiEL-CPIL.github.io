"""Layered parallel-depth construction for size-efficient sorting networks on 18 wires."""
import time
import random
import numpy as np

N = 18
DEADLINE = 82.0  # safety margin before 90s kill

# EVOLVE-BLOCK-START

def _precompute_targets():
    """Precompute expected sorted states for all 2^N boolean inputs."""
    targets = np.zeros(1 << N, dtype=np.int32)
    for mask in range(1 << N):
        ones = bin(mask).count('1')
        if ones == 0:
            targets[mask] = 0
        elif ones == N:
            targets[mask] = (1 << N) - 1
        else:
            targets[mask] = ((1 << ones) - 1) << (N - ones)
    return targets

def _simulate_network_bitparallel(net, targets):
    """
    Simulate a network on all 2^N inputs using bit-parallel NumPy.
    Returns the number of unsorted inputs (0 means valid sorting network).
    """
    if len(net) == 0:
        if N <= 1:
            return 0
        return (1 << N) - 1
    
    inputs = np.arange(1 << N, dtype=np.int32)
    
    for i, j in net:
        bit_i = (inputs >> i) & 1
        bit_j = (inputs >> j) & 1
        cond = (bit_i == 1) & (bit_j == 0)
        mask_i = 1 << i
        mask_j = 1 << j
        inputs = np.where(cond, (inputs & ~mask_i) | mask_j, inputs)
    
    unsorted = np.sum(inputs != targets)
    return int(unsorted)

def _simulate_network_sampled(net, num_samples=5000, rng=None):
    """
    Sampled simulation for quick fitness evaluation.
    Returns the number of unsorted sampled inputs.
    """
    if rng is None:
        rng = random.Random(42)
    
    unsorted_count = 0
    for _ in range(num_samples):
        mask = rng.getrandbits(N)
        state = mask
        for i, j in net:
            if (state & (1 << i)) and not (state & (1 << j)):
                state = (state & ~(1 << i)) | (1 << j)
        
        ones = bin(state).count('1')
        if ones == 0:
            expected = 0
        elif ones == N:
            expected = (1 << N) - 1
        else:
            expected = ((1 << ones) - 1) << (N - ones)
        
        if state != expected:
            unsorted_count += 1
    
    return unsorted_count

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net

def _bitonic_sort_network():
    """Generate a bitonic sort network."""
    def bitonic_merge(start, length, direction):
        comps = []
        if length == 1:
            return comps
        half = length // 2
        for i in range(half):
            if direction == 'up':
                comps.append((start + i, start + i + half))
            else:
                comps.append((start + i + half, start + i))
        comps.extend(bitonic_merge(start, half, 'up'))
        comps.extend(bitonic_merge(start + half, half, 'down'))
        return comps
    
    def bitonic_sort(start, length, direction):
        comps = []
        if length <= 1:
            return comps
        half = length // 2
        comps.extend(bitonic_sort(start, half, 'up'))
        comps.extend(bitonic_sort(start + half, half, 'down'))
        comps.extend(bitonic_merge(start, length, direction))
        return comps
    
    return bitonic_sort(0, N, 'up')

def _get_valid_layer_candidates():
    """
    Get all possible valid parallel layers (sets of non-conflicting comparators).
    A valid layer has no two comparators sharing a wire.
    Returns a list of layers, each layer is a frozenset of (i,j) pairs.
    """
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # For each wire, which comparators use it
    wire_to_comps = {w: [] for w in range(N)}
    for comp in all_comps:
        wire_to_comps[comp[0]].append(comp)
        wire_to_comps[comp[1]].append(comp)
    
    # Generate all valid layers by selecting a matching in the comparator graph
    # Use a simpler approach: enumerate subsets of comparators that form a matching
    # This is expensive for N=18, so we'll use a greedy heuristic instead
    
    # For now, return a set of common layer patterns
    layers = []
    
    # Pattern 1: Adjacent comparators (odd or even)
    for parity in [0, 1]:
        layer = frozenset((i, i+1) for i in range(parity, N-1, 2))
        if layer:
            layers.append(layer)
    
    # Pattern 2: Stride-2 comparators
    for start in range(0, 4):
        layer = set()
        for i in range(start, N-1, 2):
            if i+1 < N:
                layer.add((i, i+1))
        if layer:
            layers.append(frozenset(layer))
    
    # Pattern 3: Cross comparators
    for offset in range(1, 5):
        layer = set()
        for i in range(N - offset):
            layer.add((i, i+offset))
        if layer:
            layers.append(frozenset(layer))
    
    # Pattern 4: Random matchings
    rng = random.Random(12345)
    for _ in range(50):
        layer = set()
        used = set()
        for i in range(N):
            if i in used:
                continue
            possible_js = [j for j in range(i+1, N) if j not in used]
            if possible_js:
                j = rng.choice(possible_js)
                layer.add((i, j))
                used.add(i)
                used.add(j)
        if layer:
            layers.append(frozenset(layer))
    
    return layers

def _layered_network_to_flat(layers):
    """Convert a list of layers to a flat list of comparators."""
    net = []
    for layer in layers:
        net.extend(sorted(layer))
    return net

def _flat_network_to_layers(net):
    """
    Convert a flat network to a list of layers.
    Each layer is a maximal set of non-conflicting comparators from consecutive positions.
    """
    if not net:
        return []
    
    layers = []
    current_layer = set()
    used_wires = set()
    
    for comp in net:
        i, j = comp
        if i not in used_wires and j not in used_wires:
            current_layer.add(comp)
            used_wires.add(i)
            used_wires.add(j)
        else:
            if current_layer:
                layers.append(frozenset(current_layer))
            current_layer = set()
            used_wires = set()
            current_layer.add(comp)
            used_wires.add(i)
            used_wires.add(j)
    
    if current_layer:
        layers.append(frozenset(current_layer))
    
    return layers

def _evaluate_layered_network(layers, targets):
    """Evaluate a layered network (list of layers) for validity and size."""
    net = _layered_network_to_flat(layers)
    unsorted = _simulate_network_bitparallel(net, targets)
    return unsorted, len(net)

def _greedy_layer_construction(targets, deadline, rng):
    """
    Greedily construct a layered sorting network.
    Start with an empty state for all inputs, and at each step add a layer
    that maximally reduces the number of unsorted inputs.
    """
    # Track state for all inputs
    # state[k] = current N-bit state for input k
    inputs = np.arange(1 << N, dtype=np.int32)
    targets_arr = targets
    
    layers = []
    total_comparators = 0
    
    while time.time() < deadline:
        # Check if all sorted
        if np.all(inputs == targets_arr):
            break
        
        # Try each candidate layer and pick the one that reduces unsorted count most
        candidates = _get_valid_layer_candidates()
        rng.shuffle(candidates)
        
        best_layer = None
        best_unsorted = np.sum(inputs != targets_arr)
        best_delta = 0
        
        for layer in candidates:
            if not layer:
                continue
            
            # Simulate this layer on all inputs
            temp_inputs = inputs.copy()
            for i, j in sorted(layer):
                bit_i = (temp_inputs >> i) & 1
                bit_j = (temp_inputs >> j) & 1
                cond = (bit_i == 1) & (bit_j == 0)
                mask_i = 1 << i
                mask_j = 1 << j
                temp_inputs = np.where(cond, (temp_inputs & ~mask_i) | mask_j, temp_inputs)
            
            unsorted_after = np.sum(temp_inputs != targets_arr)
            delta = best_unsorted - unsorted_after
            
            if delta > best_delta:
                best_delta = delta
                best_layer = layer
                best_unsorted = unsorted_after
        
        if best_layer is None or best_delta <= 0:
            # No improvement possible with current candidates
            # Try adding a random comparator
            i = rng.randint(0, N-2)
            j = rng.randint(i+1, N-1)
            layer = frozenset([(i, j)])
            
            temp_inputs = inputs.copy()
            bit_i = (temp_inputs >> i) & 1
            bit_j = (temp_inputs >> j) & 1
            cond = (bit_i == 1) & (bit_j == 0)
            mask_i = 1 << i
            mask_j = 1 << j
            temp_inputs = np.where(cond, (temp_inputs & ~mask_i) | mask_j, temp_inputs)
            
            unsorted_after = np.sum(temp_inputs != targets_arr)
            if unsorted_after < best_unsorted:
                best_layer = layer
                best_unsorted = unsorted_after
                best_delta = best_unsorted - unsorted_after
            else:
                # Give up
                break
        
        if best_layer is not None:
            layers.append(best_layer)
            # Apply the layer to inputs
            for i, j in sorted(best_layer):
                bit_i = (inputs >> i) & 1
                bit_j = (inputs >> j) & 1
                cond = (bit_i == 1) & (bit_j == 0)
                mask_i = 1 << i
                mask_j = 1 << j
                inputs = np.where(cond, (inputs & ~mask_i) | mask_j, inputs)
            total_comparators += len(best_layer)
        
        if total_comparators > 4096:
            break
    
    return layers

def _optimize_layered_network(layers, targets, deadline, rng):
    """
    Optimize a layered network by:
    1. Removing redundant layers
    2. Swapping comparators within layers
    3. Merging adjacent compatible layers
    """
    current_layers = [frozenset(layer) for layer in layers]
    best_layers = list(current_layers)
    best_size = sum(len(layer) for layer in current_layers)
    
    # Validate initial
    net = _layered_network_to_flat(current_layers)
    unsorted = _simulate_network_bitparallel(net, targets)
    if unsorted != 0:
        return best_layers, best_size
    
    while time.time() < deadline:
        improved = False
        
        # Try removing each layer
        for idx in range(len(current_layers)):
            if time.time() > deadline:
                break
            
            candidate = current_layers[:idx] + current_layers[idx+1:]
            if not candidate:
                continue
            
            net = _layered_network_to_flat(candidate)
            unsorted = _simulate_network_bitparallel(net, targets)
            if unsorted == 0:
                current_layers = candidate
                new_size = sum(len(layer) for layer in current_layers)
                if new_size < best_size:
                    best_layers = list(current_layers)
                    best_size = new_size
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping comparators between adjacent layers
        for idx in range(len(current_layers) - 1):
            if time.time() > deadline:
                break
            
            layer1 = set(current_layers[idx])
            layer2 = set(current_layers[idx+1])
            
            # Try moving a comparator from layer1 to layer2 or vice versa
            for comp in list(layer1):
                i, j = comp
                # Check if comp can go to layer2 (no wire conflict)
                can_move = True
                for other_i, other_j in layer2:
                    if i in (other_i, other_j) or j in (other_i, other_j):
                        can_move = False
                        break
                
                if can_move:
                    new_layer1 = layer1 - {comp}
                    new_layer2 = layer2 | {comp}
                    
                    candidate = current_layers[:idx] + [frozenset(new_layer1), frozenset(new_layer2)] + current_layers[idx+2:]
                    # Remove empty layers
                    candidate = [layer for layer in candidate if layer]
                    
                    if candidate:
                        net = _layered_network_to_flat(candidate)
                        unsorted = _simulate_network_bitparallel(net, targets)
                        if unsorted == 0:
                            current_layers = candidate
                            new_size = sum(len(layer) for layer in current_layers)
                            if new_size < best_size:
                                best_layers = list(current_layers)
                                best_size = new_size
                            improved = True
                            break
            
            if improved:
                break
        
        if not improved:
            # Try random perturbation
            if len(current_layers) > 0:
                idx = rng.randint(0, len(current_layers) - 1)
                layer = list(current_layers[idx])
                if layer:
                    comp = rng.choice(layer)
                    i, j = comp
                    # Replace with a random comparator
                    new_i = rng.randint(0, N-2)
                    new_j = rng.randint(new_i+1, N-1)
                    new_comp = (new_i, new_j)
                    
                    # Check for wire conflicts within the layer
                    can_add = True
                    for other_i, other_j in layer:
                        if other_i == comp:
                            continue
                        if new_i in (other_i, other_j) or new_j in (other_i, other_j):
                            can_add = False
                            break
                    
                    if can_add:
                        new_layer = (set(layer) - {comp}) | {new_comp}
                        candidate = current_layers[:idx] + [frozenset(new_layer)] + current_layers[idx+1:]
                        
                        net = _layered_network_to_flat(candidate)
                        unsorted = _simulate_network_bitparallel(net, targets)
                        if unsorted == 0:
                            current_layers = candidate
                            new_size = sum(len(layer) for layer in current_layers)
                            if new_size < best_size:
                                best_layers = list(current_layers)
                                best_size = new_size
                            improved = True
        
        if not improved:
            # Try adding a new layer with one random comparator
            if len(current_layers) < 200:
                i = rng.randint(0, N-2)
                j = rng.randint(i+1, N-1)
                new_layer = frozenset([(i, j)])
                
                # Try inserting at different positions
                for pos in range(len(current_layers) + 1):
                    if time.time() > deadline:
                        break
                    
                    candidate = current_layers[:pos] + [new_layer] + current_layers[pos:]
                    net = _layered_network_to_flat(candidate)
                    unsorted = _simulate_network_bitparallel(net, targets)
                    if unsorted == 0:
                        current_layers = candidate
                        new_size = sum(len(layer) for layer in current_layers)
                        if new_size < best_size:
                            best_layers = list(current_layers)
                            best_size = new_size
                        improved = True
                        break
                
                if improved:
                    continue
        
        if not improved:
            break
    
    return best_layers, best_size

def _improve_by_removal(net, targets, deadline):
    """
    Try to improve a network by removing redundant comparators.
    """
    current = list(net)
    best = list(current)
    best_len = len(current)
    
    unsorted = _simulate_network_bitparallel(current, targets)
    if unsorted != 0:
        return current
    
    rng = random.Random(123)
    
    while time.time() < deadline:
        if not current:
            break
        
        indices = list(range(len(current)))
        rng.shuffle(indices)
        
        improved = False
        for idx in indices:
            if time.time() > deadline:
                break
            
            candidate = current[:idx] + current[idx+1:]
            
            sampled_unsorted = _simulate_network_sampled(candidate, num_samples=2000, rng=rng)
            if sampled_unsorted > 0:
                continue
            
            unsorted = _simulate_network_bitparallel(candidate, targets)
            if unsorted == 0:
                current = candidate
                best = list(current)
                best_len = len(current)
                improved = True
                break
        
        if not improved:
            for _ in range(5):
                if time.time() > deadline:
                    break
                new_net = list(current)
                if len(new_net) >= 2:
                    i, j = rng.sample(range(len(new_net)), 2)
                    new_net[i], new_net[j] = new_net[j], new_net[i]
                    sampled_unsorted = _simulate_network_sampled(new_net, num_samples=2000, rng=rng)
                    if sampled_unsorted == 0:
                        unsorted = _simulate_network_bitparallel(new_net, targets)
                        if unsorted == 0:
                            current = new_net
                            if len(current) < best_len:
                                best = list(current)
                                best_len = len(current)
                            improved = True
                            break
    
    return best

def construct():
    start_time = time.time()
    
    # Precompute targets
    targets = _precompute_targets()
    
    # Start with the seed network
    seed_net = _odd_even_transposition()
    
    # Verify seed
    unsorted = _simulate_network_bitparallel(seed_net, targets)
    if unsorted != 0:
        raise ValueError("Seed network is invalid")
    
    best_net = list(seed_net)
    best_len = len(seed_net)
    
    rng = random.Random(42)
    
    # Strategy 1: Greedy layered construction
    if time.time() < DEADLINE - 20:
        greedy_deadline = time.time() + min(30, DEADLINE - time.time() - 10)
        greedy_layers = _greedy_layer_construction(targets, greedy_deadline, rng)
        if greedy_layers:
            greedy_net = _layered_network_to_flat(greedy_layers)
            unsorted = _simulate_network_bitparallel(greedy_net, targets)
            if unsorted == 0 and len(greedy_net) < best_len:
                best_net = greedy_net
                best_len = len(greedy_net)
            
            # Optimize the greedy result
            if time.time() < DEADLINE - 15:
                opt_deadline = time.time() + min(15, DEADLINE - time.time() - 5)
                opt_layers, opt_size = _optimize_layered_network(greedy_layers, targets, opt_deadline, rng)
                opt_net = _layered_network_to_flat(opt_layers)
                unsorted = _simulate_network_bitparallel(opt_net, targets)
                if unsorted == 0 and len(opt_net) < best_len:
                    best_net = opt_net
                    best_len = len(opt_net)
    
    # Strategy 2: Try bitonic sort network
    if time.time() < DEADLINE - 15:
        bitonic = _bitonic_sort_network()
        unsorted = _simulate_network_bitparallel(bitonic, targets)
        if unsorted == 0 and len(bitonic) < best_len:
            best_net = list(bitonic)
            best_len = len(bitonic)
        
        # Convert to layers and optimize
        if time.time() < DEADLINE - 10:
            bitonic_layers = _flat_network_to_layers(bitonic)
            opt_deadline = time.time() + min(10, DEADLINE - time.time() - 5)
            opt_layers, opt_size = _optimize_layered_network(bitonic_layers, targets, opt_deadline, rng)
            opt_net = _layered_network_to_flat(opt_layers)
            unsorted = _simulate_network_bitparallel(opt_net, targets)
            if unsorted == 0 and len(opt_net) < best_len:
                best_net = opt_net
                best_len = len(opt_net)
    
    # Strategy 3: Multiple greedy restarts with different random seeds
    if time.time() < DEADLINE - 10:
        for seed in [1, 2, 3]:
            if time.time() > DEADLINE - 8:
                break
            rng_restart = random.Random(seed)
            greedy_deadline = time.time() + min(5, DEADLINE - time.time() - 3)
            greedy_layers = _greedy_layer_construction(targets, greedy_deadline, rng_restart)
            if greedy_layers:
                greedy_net = _layered_network_to_flat(greedy_layers)
                unsorted = _simulate_network_bitparallel(greedy_net, targets)
                if unsorted == 0 and len(greedy_net) < best_len:
                    best_net = greedy_net
                    best_len = len(greedy_net)
    
    # Strategy 4: Final improvement on best network
    if time.time() < DEADLINE - 3:
        improved = _improve_by_removal(best_net, targets, deadline=time.time() + min(3, DEADLINE - time.time() - 2))
        if len(improved) < best_len:
            best_net = improved
            best_len = len(improved)
    
    # Final verification
    unsorted = _simulate_network_bitparallel(best_net, targets)
    if unsorted != 0:
        best_net = seed_net
    
    if len(best_net) > 4096:
        best_net = best_net[:4096]
    
    return best_net

# EVOLVE-BLOCK-END
