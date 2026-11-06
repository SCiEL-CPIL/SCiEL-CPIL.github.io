"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17

# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _all_bool_inputs(n):
    """Generate all 2^n boolean inputs as a (2^n, n) array of 0s and 1s."""
    return np.array([[int((mask >> i) & 1) for i in range(n)] for mask in range(1 << n)], dtype=np.int8)

def _is_sorted_network(comparators, inputs):
    """Check if the network sorts all given inputs using zero-one principle."""
    n = inputs.shape[1]
    # Process all inputs in parallel
    state = inputs.copy()
    for (i, j) in comparators:
        # compare-exchange: min to i, max to j
        col_i = state[:, i]
        col_j = state[:, j]
        new_i = np.minimum(col_i, col_j)
        new_j = np.maximum(col_i, col_j)
        state[:, i] = new_i
        state[:, j] = new_j
    # Check if all outputs are sorted (non-decreasing)
    # For a sorted array, each element should be >= previous
    for k in range(1, n):
        if np.any(state[:, k] < state[:, k - 1]):
            return False
    return True

def _seed_network():
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires (n must be power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively sort two halves
    left = _bitonic_sort_network(half)
    right = [(i + half, j + half) for (i, j) in left]
    # Merge: compare elements i and i+half for i in 0..half-1
    merge = [(i, i + half) for i in range(half)]
    return left + right + merge

def _pad_to_power_of_2(network, original_n):
    """Pad a network for power-of-2 wires to work with original_n wires."""
    pass

def _construct_bitonic_for_17():
    """
    Construct a bitonic-like sorting network for 17 wires.
    Uses the fact that 17 = 16 + 1. We can sort 16 wires with bitonic,
    then merge the 17th wire in.
    """
    # First, create a bitonic network for 16 wires (wires 0-15)
    net16 = _bitonic_sort_network(16)
    
    # Now we need to merge wire 16 into the sorted 16 wires.
    # Strategy: compare wire 16 with each of wires 0-15 in a tree-like fashion
    # to find its correct position, then shift wires.
    
    # A simpler approach: just append comparators to merge wire 16
    # Compare 16 with 15, then 15 with 14, etc. (like insertion sort)
    # But this is inefficient. Better: use a tournament tree.
    
    # Let's use a different approach: 
    # 1. Sort wires 0-15 with bitonic (16 wires)
    # 2. Then do a merge of the sorted 16 with wire 16
    
    # For merging a single element into a sorted array of 16:
    # We can use a binary search tree of comparisons.
    # Compare 16 with 8 (middle of 0-15)
    # If 16 < 8, then compare 16 with 4, then 2, then 1, then 0
    # If 16 > 8, then compare 16 with 12, then 14, then 15
    # But we also need to shift the other wires.
    
    # Actually, the cleanest approach for 17:
    # Use the fact that we can sort 17 by:
    # 1. Sort 0-14 (15 wires) - but 15 is not power of 2
    # 2. Alternative: sort 0-15 (16 wires) with bitonic, then insert 16
    
    # Let me just use odd-even transposition as a better seed and focus on optimization
    return _seed_network()

def _simulate_network_layers(layers, inputs):
    """Simulate a network given as layers of non-overlapping comparators."""
    n = inputs.shape[1]
    state = inputs.copy()
    for layer in layers:
        for (i, j) in layer:
            col_i = state[:, i]
            col_j = state[:, j]
            new_i = np.minimum(col_i, col_j)
            new_j = np.maximum(col_i, col_j)
            state[:, i] = new_i
            state[:, j] = new_j
    return state

def _is_sorted_network_layers(layers, inputs):
    """Check if the network (as layers) sorts all given inputs."""
    n = inputs.shape[1]
    state = _simulate_network_layers(layers, inputs)
    for k in range(1, n):
        if np.any(state[:, k] < state[:, k - 1]):
            return False
    return True

def _comparators_to_layers(comparators):
    """Convert a list of comparators to layers (greedy: group non-overlapping)."""
    layers = []
    used = [False] * N
    for (i, j) in comparators:
        # Find a layer where both i and j are free
        placed = False
        for layer in layers:
            if not used[i] and not used[j]:
                # Check if i and j are free in this layer
                wires_in_layer = set()
                for (a, b) in layer:
                    wires_in_layer.add(a)
                    wires_in_layer.add(b)
                if i not in wires_in_layer and j not in wires_in_layer:
                    layer.append((i, j))
                    placed = True
                    break
        if not placed:
            layers.append([(i, j)])
    return layers

def _layers_to_comparators(layers):
    """Convert layers back to a flat list of comparators."""
    comparators = []
    for layer in layers:
        comparators.extend(layer)
    return comparators

def construct():
    deadline = time.time() + 80  # Safety margin before 90s limit
    start_time = time.time()
    
    # Pre-compute all boolean inputs
    all_inputs = _all_bool_inputs(N)
    
    # Start with seed network
    network = _seed_network()
    
    # Convert to layers representation
    layers = _comparators_to_layers(network)
    
    # Track best
    best_network = network
    best_count = len(network)
    
    # Phase 1: Aggressive removal using layers
    # Try removing each comparator, check validity
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(network)):
            if time.time() >= deadline:
                break
            # Try removing comparator at idx
            candidate = network[:idx] + network[idx+1:]
            if _is_sorted_network(candidate, all_inputs):
                network = candidate
                best_network = candidate
                best_count = len(network)
                improved = True
                break  # Restart from beginning since indices shifted
    
    # Phase 2: Try swapping adjacent comparators
    # This can help if the order matters for efficiency
    if time.time() < deadline:
        for _ in range(100):  # Limited iterations
            if time.time() >= deadline:
                break
            if len(network) < 2:
                break
            idx = random.randint(0, len(network) - 2)
            candidate = network[:]
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if _is_sorted_network(candidate, all_inputs):
                # Check if this swap reduces depth or is otherwise beneficial
                # For now, accept if it doesn't increase count
                network = candidate
                best_network = candidate
                best_count = len(network)
    
    # Phase 3: Try replacing pairs of comparators with single comparators
    # If two comparators (i,j) and (k,l) can be replaced by (i,l) and (k,j) or similar
    if time.time() < deadline:
        for _ in range(50):
            if time.time() >= deadline:
                break
            if len(network) < 2:
                break
            idx1 = random.randint(0, len(network) - 1)
            idx2 = random.randint(0, len(network) - 1)
            if idx1 == idx2:
                continue
            i1, j1 = network[idx1]
            i2, j2 = network[idx2]
            # Try alternative pairings
            alternatives = [
                [(i1, j2), (i2, j1)],
                [(i1, i2), (j1, j2)],
            ]
            for alt in alternatives:
                # Ensure valid pairs
                valid = True
                for (a, b) in alt:
                    if a >= b or a < 0 or b >= N:
                        valid = False
                        break
                if not valid:
                    continue
                candidate = network[:]
                # Replace idx1 and idx2 (handle ordering)
                if idx1 < idx2:
                    candidate[idx1] = alt[0]
                    candidate[idx2] = alt[1]
                else:
                    candidate[idx2] = alt[0]
                    candidate[idx1] = alt[1]
                if _is_sorted_network(candidate, all_inputs):
                    network = candidate
                    best_network = candidate
                    best_count = len(network)
                    break
    
    return best_network
# EVOLVE-BLOCK-END
