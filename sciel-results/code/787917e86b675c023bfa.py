"""Combined program for sorting networks on 17 wires.
Uses a layered representation and Simulated Annealing for structural optimization.
"""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute the input matrix and sorted reference once.
# Shape: (2^N, N)
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses pre-computed _INPUTS and _SORTED_INPUTS for speed.
    """
    if len(comparators) > 4096:
        return False
    
    if not comparators:
        return False

    # Copy the input matrix to avoid modifying the global constant
    state = _INPUTS.copy()
    
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return np.array_equal(state, _SORTED_INPUTS)

def layers_to_flat(layers):
    """Convert a list of layers (lists of tuples) to a flat list of tuples."""
    flat = []
    for layer in layers:
        flat.extend(layer)
    return flat

def flat_to_layers(flat):
    """
    Convert a flat list of comparators to a list of layers.
    This is a greedy assignment: assign each comparator to the earliest layer
    where it doesn't conflict with existing comparators.
    """
    layers = []
    for comp in flat:
        i, j = comp
        placed = False
        for l_idx, layer in enumerate(layers):
            # Check if i or j is used in this layer
            conflict = False
            for a, b in layer:
                if i in (a, b) or j in (a, b):
                    conflict = True
                    break
            if not conflict:
                layer.append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return layers

def construct():
    start_time = time.time()
    # Safety margin: aim to finish well before 60s
    deadline = start_time + 55.0
    
    # 1. Generate Seed Network (Odd-Even Transposition)
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
            
    # Verify seed
    if not is_sorted_network(seed_network):
        return seed_network

    best_network = seed_network
    best_len = len(seed_network)
    
    # Initialize search state with layered representation
    current_layers = flat_to_layers(seed_network)
    
    # Simulated Annealing Parameters
    T = 1.0
    T_min = 0.01
    alpha = 0.9995 # Cooling rate
    
    # We will iterate while time permits and T > T_min
    while time.time() < deadline and T > T_min:
        # Generate a random move
        move_type = random.choice(['remove', 'add', 'swap', 'move'])
        new_layers = [list(layer) for layer in current_layers]
        
        valid_move = True
        
        if move_type == 'remove' and current_layers:
            # Remove a random comparator from a random non-empty layer
            non_empty_layers = [i for i, l in enumerate(new_layers) if l]
            if non_empty_layers:
                l_idx = random.choice(non_empty_layers)
                c_idx = random.randint(0, len(new_layers[l_idx]) - 1)
                new_layers[l_idx].pop(c_idx)
                # Clean up empty layers? Optional, but keeps structure clean
                if not new_layers[l_idx]:
                    new_layers.pop(l_idx)
            else:
                valid_move = False
                
        elif move_type == 'add':
            # Add a random comparator to a random layer (or new layer)
            # Generate random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            new_comp = (i, j)
            
            if not new_layers:
                new_layers.append([new_comp])
            else:
                # Try to add to an existing layer
                l_idx = random.randint(0, len(new_layers) - 1)
                conflict = False
                for a, b in new_layers[l_idx]:
                    if i in (a, b) or j in (a, b):
                        conflict = True
                        break
                if not conflict:
                    new_layers[l_idx].append(new_comp)
                else:
                    # Add to a new layer
                    new_layers.append([new_comp])
                    
        elif move_type == 'swap' and len(current_layers) >= 2:
            # Swap comparators between two different layers
            l1 = random.randint(0, len(new_layers) - 1)
            l2 = random.randint(0, len(new_layers) - 1)
            if l1 == l2:
                valid_move = False
            elif not new_layers[l1] or not new_layers[l2]:
                valid_move = False
            else:
                c1_idx = random.randint(0, len(new_layers[l1]) - 1)
                c2_idx = random.randint(0, len(new_layers[l2]) - 1)
                c1 = new_layers[l1][c1_idx]
                c2 = new_layers[l2][c2_idx]
                
                # Check if swap is valid (no conflicts in new positions)
                # Remove c1 from l1 and c2 from l2
                new_layers[l1][c1_idx] = c2
                new_layers[l2][c2_idx] = c1
                
                # Verify no conflicts in l1 with c2
                for a, b in new_layers[l1]:
                    if a == c2[0] or a == c2[1] or b == c2[0] or b == c2[1]:
                        if (a, b) != c2:
                            valid_move = False
                            break
                # Verify no conflicts in l2 with c1
                if valid_move:
                    for a, b in new_layers[l2]:
                        if a == c1[0] or a == c1[1] or b == c1[0] or b == c1[1]:
                            if (a, b) != c1:
                                valid_move = False
                                break
                                
                if not valid_move:
                    # Revert
                    new_layers[l1][c1_idx] = c1
                    new_layers[l2][c2_idx] = c2

        elif move_type == 'move' and len(current_layers) >= 1:
            # Move a comparator to a different layer
            if not new_layers:
                valid_move = False
            else:
                l_from = random.randint(0, len(new_layers) - 1)
                if not new_layers[l_from]:
                    valid_move = False
                else:
                    c_idx = random.randint(0, len(new_layers[l_from]) - 1)
                    comp = new_layers[l_from][c_idx]
                    i, j = comp
                    
                    # Choose a target layer (existing or new)
                    if random.random() < 0.8 and len(new_layers) > 1:
                        l_to = random.randint(0, len(new_layers) - 1)
                        if l_to == l_from:
                            valid_move = False
                        else:
                            # Check conflict in target
                            conflict = False
                            for a, b in new_layers[l_to]:
                                if i in (a, b) or j in (a, b):
                                    conflict = True
                                    break
                            if not conflict:
                                new_layers[l_from].pop(c_idx)
                                if not new_layers[l_from]:
                                    new_layers.pop(l_from)
                                    # Adjust l_to if it was after l_from
                                    if l_to > l_from:
                                        l_to -= 1
                                new_layers[l_to].append(comp)
                            else:
                                valid_move = False
                    else:
                        # Move to a new layer
                        new_layers[l_from].pop(c_idx)
                        if not new_layers[l_from]:
                            new_layers.pop(l_from)
                        new_layers.append([comp])

        if not valid_move:
            continue
            
        # Calculate cost
        new_flat = layers_to_flat(new_layers)
        new_len = len(new_flat)
        old_len = sum(len(l) for l in current_layers)
        
        # Check validity (expensive)
        if is_sorted_network(new_flat):
            # Accept if better, or with probability exp(-delta/T)
            delta = new_len - old_len
            if delta < 0:
                accept = True
            else:
                accept = random.random() < np.exp(-delta / T)
                
            if accept:
                current_layers = new_layers
                if new_len < best_len:
                    best_len = new_len
                    best_network = new_flat
        else:
            # If invalid, we generally reject. 
            # In SA, we might accept invalid states if we had a "soft" penalty, 
            # but here validity is a hard constraint for the final output.
            # However, for search, it's often better to stay in the valid space 
            # or use a penalty. Given the cost of checking, we just reject.
            pass

        T *= alpha

    # Final check
    if is_sorted_network(best_network):
        return best_network
    return seed_network
