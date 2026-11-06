"""Size-efficient sorting network on 17 wires using layer-based representation."""
import numpy as np
from itertools import product

N = 17

# EVOLVE-BLOCK-START
def _apply_layer(wires, layer):
    """Apply a layer of disjoint comparators to the wires array.
    
    Args:
        wires: numpy array of shape (num_inputs, N) or (N,)
        layer: list of (i, j) pairs, all disjoint (no shared indices)
    
    Returns:
        New wires array after applying all comparators in the layer.
    """
    if wires.ndim == 1:
        wires = wires.copy()
        for i, j in layer:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        return wires
    else:
        new_wires = wires.copy()
        for i, j in layer:
            mask = new_wires[:, i] > new_wires[:, j]
            new_wires[:, i] = np.where(mask, new_wires[:, j], new_wires[:, i])
            new_wires[:, j] = np.where(mask, new_wires[:, i], new_wires[:, j])
        return new_wires

def _is_sorted(wires):
    """Check if all rows of wires are sorted in non-decreasing order."""
    if wires.ndim == 1:
        return np.all(np.diff(wires) >= 0)
    else:
        return np.all(np.diff(wires, axis=1) >= 0)

def _verify_network(layers):
    """Verify that the network sorts all 2^17 Boolean inputs."""
    # Generate all 2^17 Boolean inputs
    num_inputs = 1 << N
    # Use vectorized approach: generate all inputs as a (num_inputs, N) array
    # For N=17, that's 131072 x 17 which is manageable
    inputs = np.zeros((num_inputs, N), dtype=np.int8)
    for i in range(N):
        # Bit i of each row
        inputs[:, i] = (np.arange(num_inputs) >> i) & 1
    
    # Apply each layer
    wires = inputs
    for layer in layers:
        wires = _apply_layer(wires, layer)
    
    # Check if all rows are sorted
    return _is_sorted(wires)

def _count_comparators(layers):
    """Count total number of comparators in the network."""
    return sum(len(layer) for layer in layers)

def _get_depth(layers):
    """Get the depth (number of layers) of the network."""
    return len(layers)

def construct():
    """Construct a sorting network on 17 wires.
    
    Uses a layer-based representation where each layer contains disjoint
    comparators applied simultaneously. We start from a known good baseline
    and attempt to optimize via local search.
    
    Returns:
        List of (i, j) integer pairs representing the comparators in order.
    """
    # Start with odd-even transposition as a baseline, but reorganize into layers
    # Odd-even transposition: N phases, each phase has disjoint comparators
    # Phase 0: (0,1), (2,3), (4,5), ...
    # Phase 1: (1,2), (3,4), (5,6), ...
    # etc.
    
    # Build the odd-even transposition network as layers
    base_layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        base_layers.append(layer)
    
    # Verify baseline
    if not _verify_network(base_layers):
        # Fallback: return flat list from odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Now try to optimize: remove redundant comparators
    # Strategy: for each comparator, check if removing it still yields a valid network
    # This is expensive but we can do it with the zero-one principle verification
    
    layers = [list(layer) for layer in base_layers]
    current_count = _count_comparators(layers)
    
    # Try to remove each comparator one by one (greedy removal)
    # We'll iterate through all comparators and try to remove them
    improved = True
    max_iterations = 50  # Limit iterations to stay within time budget
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try removing each comparator in each layer
        for layer_idx in range(len(layers)):
            for comp_idx in range(len(layers[layer_idx])):
                # Temporarily remove the comparator
                saved = layers[layer_idx].pop(comp_idx)
                
                # Check if still valid
                if _verify_network(layers):
                    improved = True
                    current_count -= 1
                    # Don't re-add it
                else:
                    # Re-add it
                    layers[layer_idx].insert(comp_idx, saved)
                
                # If we found an improvement, restart the scan
                if improved:
                    break
            
            if improved:
                break
    
    # Also try merging adjacent layers where possible (if comparators are disjoint)
    # and removing empty layers
    new_layers = []
    for layer in layers:
        if not layer:
            continue
        # Check if this layer can be merged with the previous one
        if new_layers:
            prev = new_layers[-1]
            # Check if all indices are disjoint between prev and layer
            prev_indices = set()
            for i, j in prev:
                prev_indices.add(i)
                prev_indices.add(j)
            layer_indices = set()
            for i, j in layer:
                layer_indices.add(i)
                layer_indices.add(j)
            if prev_indices.isdisjoint(layer_indices):
                # Merge
                new_layers[-1] = prev + layer
                continue
        new_layers.append(list(layer))
    
    layers = new_layers
    
    # Final verification
    if _verify_network(layers):
        # Flatten the layers into a single list of comparators
        result = []
        for layer in layers:
            result.extend(layer)
        return result
    else:
        # Fallback to original
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
