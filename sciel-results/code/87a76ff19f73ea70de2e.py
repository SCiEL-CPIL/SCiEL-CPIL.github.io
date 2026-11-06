"""Size-efficient sorting network synthesis for 18 wires.

Structural approach: Represents the network as a sequence of parallel layers.
Uses Batcher's Odd-Even Mergesort as a strong initial seed.
Optimizes via layer-wise greedy removal, layer merging, and genetic mutations
on the layer structure.
"""
import time
import random

N = 18
_M = 1 << N  # 262144

# Precompute initial wire states as big integers.
_wires_init = [0] * N
for w in range(N):
    mask = 0
    for k in range(_M):
        if (k >> w) & 1:
            mask |= (1 << k)
    _wires_init[w] = mask


def _verify_network_bit_parallel(comparators):
    """Return True iff the comparator list sorts all 2**N Boolean inputs."""
    if not comparators:
        return N <= 1

    state = _wires_init[:]

    for i, j in comparators:
        swap_mask = state[i] & ~state[j]
        if swap_mask == 0:
            continue
        old_i = state[i]
        old_j = state[j]
        state[i] = (old_i & ~swap_mask) | (old_j & swap_mask)
        state[j] = (old_j & ~swap_mask) | (old_i & swap_mask)

    for w in range(N - 1):
        if state[w] & ~state[w + 1]:
            return False
    return True


def _batcher_odd_even_merge(n):
    """Generate comparators for Batcher's Odd-Even Merge for n elements (power of 2)."""
    if n <= 1:
        return []
    
    # Recursive split
    half = n // 2
    # Merge two sorted halves of size half
    # Batcher's merge: 
    # 1. Recursively merge odd and even indexed elements (indices 1,3.. and 0,2..)
    # 2. Compare-exchange adjacent pairs
    
    # Standard Batcher Odd-Even Merge implementation
    comparators = []
    
    # 1. Recursively merge the two halves (indices 0..half-1 and half..2*half-1)
    # This part is usually handled by the caller if we are building a full sort,
    # but for a merge network, we assume inputs are pre-sorted halves.
    # However, for the full network construction, we need to generate the merge stages.
    
    # Let's use the standard recursive definition for Odd-Even Merge Network
    # M(n) = M(n/2) on odds, M(n/2) on evens, then compare (i, i+1) for i=0..n-2 with step 2?
    # Actually, Batcher's Odd-Even Merge:
    # 1. Merge odd-indexed elements (1, 3, ...)
    # 2. Merge even-indexed elements (0, 2, ...)
    # 3. Compare-exchange (i, i+1) for i=0, 2, 4, ... (even-odd pairs)
    
    # Wait, the standard definition is:
    # To merge two sorted sequences of length n/2:
    # 1. Recursively merge the odd-indexed elements (1, 3, ...) of the two sequences.
    # 2. Recursively merge the even-indexed elements (0, 2, ...) of the two sequences.
    # 3. Compare-exchange adjacent pairs (0,1), (2,3), ... (n-2, n-1).
    
    # Let's implement a helper that returns the comparators for merging two sorted lists of size m.
    def _merge(m):
        if m == 1:
            return []
        
        # Split indices into odd and even
        # Odd indices: 1, 3, ..., m-1 (relative to the combined 2m list? No, relative to input 2m)
        # The input is 2m elements. 
        # Odd positions: 1, 3, ..., 2m-1
        # Even positions: 0, 2, ..., 2m-2
        
        # Recursively merge the odd subsequence (size m) and even subsequence (size m)
        # Note: The recursive calls are on size m, but the indices are mapped.
        # For odd positions: index 1 -> 0, 3 -> 1, ..., 2m-1 -> m-1
        # For even positions: index 0 -> 0, 2 -> 1, ..., 2m-2 -> m-1
        
        odd_comps = _merge(m)
        even_comps = _merge(m)
        
        # Map back to global indices
        mapped_odd = []
        for i, j in odd_comps:
            # i, j are indices in 0..m-1
            # Global odd index: 2*i + 1, 2*j + 1
            mapped_odd.append((2*i + 1, 2*j + 1))
            
        mapped_even = []
        for i, j in even_comps:
            # Global even index: 2*i, 2*j
            mapped_even.append((2*i, 2*j))
            
        # Final stage: compare (0,1), (2,3), ..., (2m-2, 2m-1)
        final_comps = [(2*i, 2*i + 1) for i in range(m)]
        
        return mapped_odd + mapped_even + final_comps

    return _merge(n // 2)


def _batcher_odd_even_mergesort(n):
    """Generate the full Batcher Odd-Even Mergesort network for n (power of 2)."""
    if n <= 1:
        return []
    
    half = n // 2
    comparators = []
    
    # 1. Recursively sort first half and second half
    # First half: indices 0..half-1
    # Second half: indices half..2*half-1
    
    left_sort = _batcher_odd_even_mergesort(half)
    for i, j in left_sort:
        comparators.append((i, j))
        
    right_sort = _batcher_odd_even_mergesort(half)
    for i, j in right_sort:
        comparators.append((i + half, j + half))
        
    # 2. Merge the two sorted halves
    merge_comps = _batcher_odd_even_merge(n)
    comparators.extend(merge_comps)
    
    return comparators


def _flatten_layers(layers):
    """Flatten a list of layers (list of lists of comparators) into a single list."""
    comparators = []
    for layer in layers:
        comparators.extend(layer)
    return comparators


def _layers_to_list(layers):
    return _flatten_layers(layers)


def _list_to_layers(comparators):
    """Convert a flat list of comparators to layers.
    Simple heuristic: Group comparators that don't share wires.
    This is a greedy coloring of the wire-intersection graph.
    """
    layers = []
    current_layer_wires = set()
    current_layer = []
    
    for comp in comparators:
        i, j = comp
        if i not in current_layer_wires and j not in current_layer_wires:
            current_layer.append(comp)
            current_layer_wires.add(i)
            current_layer_wires.add(j)
        else:
            if current_layer:
                layers.append(current_layer)
            current_layer = [comp]
            current_layer_wires = {i, j}
            
    if current_layer:
        layers.append(current_layer)
        
    return layers


def _greedy_reduce_layers(layers, deadline):
    """Try to remove comparators from layers greedily."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for l_idx in range(len(layers)):
            layer = layers[l_idx]
            for c_idx in range(len(layer)):
                if time.time() > deadline:
                    return layers
                # Try removing layer[l_idx][c_idx]
                new_layer = layer[:c_idx] + layer[c_idx+1:]
                if not new_layer:
                    # If layer becomes empty, we can remove it, but let's keep structure simple for now
                    # Actually, removing empty layers is good
                    candidate_layers = layers[:l_idx] + layers[l_idx+1:]
                    flat = _flatten_layers(candidate_layers)
                    if _verify_network_bit_parallel(flat):
                        layers = candidate_layers
                        improved = True
                        break # Restart loop as indices changed
                else:
                    candidate_layers = layers[:]
                    candidate_layers[l_idx] = new_layer
                    flat = _flatten_layers(candidate_layers)
                    if _verify_network_bit_parallel(flat):
                        layers = candidate_layers
                        improved = True
                        break # Restart loop as indices changed
                        
    return layers


def _merge_adjacent_layers(layers):
    """Try to merge adjacent layers if they are disjoint."""
    merged = True
    while merged and len(layers) > 1:
        merged = False
        for i in range(len(layers) - 1):
            wires1 = set()
            for c in layers[i]:
                wires1.add(c[0])
                wires1.add(c[1])
                
            wires2 = set()
            for c in layers[i+1]:
                wires2.add(c[0])
                wires2.add(c[1])
                
            if not wires1.intersection(wires2):
                # Can merge
                new_layer = layers[i] + layers[i+1]
                layers = layers[:i] + [new_layer] + layers[i+2:]
                merged = True
                break
    return layers


def construct():
    """Construct a compare-exchange network that sorts all 18-input sequences."""
    start_time = time.time()
    deadline = start_time + 50
    
    # N is not a power of 2, so we use the network for 32 wires and ignore the extra 14?
    # Or we can just use the flat list from a known good network.
    # Batcher for 32 is large. Let's stick to a flat list initialization from Odd-Even Transposition
    # but structure it into layers for optimization.
    
    # Actually, Odd-Even Transposition is already layer-structured.
    # Phase 0: (0,1), (2,3), ...
    # Phase 1: (1,2), (3,4), ...
    
    # Let's generate OET layers
    oet_comparators = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        if layer:
            oet_comparators.append(layer)
            
    best_layers = oet_comparators
    best_flat = _flatten_layers(best_layers)
    best_len = len(best_flat)
    
    if not _verify_network_bit_parallel(best_flat):
        return []
        
    random.seed(42)
    current_layers = [list(l) for l in oet_comparators]
    
    # Initial greedy reduction
    current_layers = _greedy_reduce_layers(current_layers, deadline)
    current_flat = _flatten_layers(current_layers)
    if len(current_flat) < best_len:
        best_flat = current_flat
        best_layers = [list(l) for l in current_layers]
        best_len = len(current_flat)
        
    # Initial layer merging
    current_layers = _merge_adjacent_layers(current_layers)
    current_flat = _flatten_layers(current_layers)
    if len(current_flat) < best_len:
        best_flat = current_flat
        best_layers = [list(l) for l in current_layers]
        best_len = len(current_flat)
        
    max_iterations = 1000000
    consecutive_failures = 0
    max_consecutive_failures = 1000
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        if best_len <= 70:
            break
        if consecutive_failures > max_consecutive_failures:
            random.seed(random.randint(0, 2**32 - 1))
            consecutive_failures = 0
            current_layers = [list(l) for l in best_layers]
            # Restart with some randomization
            # Shuffle comparators within layers?
            for l in current_layers:
                random.shuffle(l)
                
        mod_type = random.random()
        
        if mod_type < 0.5:
            # Random removal attempt
            if len(current_flat) <= 1:
                continue
            # Pick random layer and comparator
            l_idx = random.randint(0, len(current_layers) - 1)
            if not current_layers[l_idx]:
                continue
            c_idx = random.randint(0, len(current_layers[l_idx]) - 1)
            
            new_layer = current_layers[l_idx][:c_idx] + current_layers[l_idx][c_idx+1:]
            candidate_layers = current_layers[:]
            candidate_layers[l_idx] = new_layer
            
            flat = _flatten_layers(candidate_layers)
            if _verify_network_bit_parallel(flat):
                current_layers = candidate_layers
                consecutive_failures = 0
                if len(flat) < best_len:
                    best_flat = flat
                    best_layers = [list(l) for l in current_layers]
                    best_len = len(flat)
                # Greedy cascade
                current_layers = _greedy_reduce_layers(current_layers, deadline)
                current_flat = _flatten_layers(current_layers)
                if len(current_flat) < best_len:
                    best_flat = current_flat
                    best_layers = [list(l) for l in current_layers]
                    best_len = len(current_flat)
            else:
                consecutive_failures += 1
                
        elif mod_type < 0.8:
            # Swap comparators between layers (if disjoint)
            if len(current_layers) < 2:
                continue
            l1, l2 = random.sample(range(len(current_layers)), 2)
            if not current_layers[l1] or not current_layers[l2]:
                continue
            c1 = random.choice(current_layers[l1])
            c2 = random.choice(current_layers[l2])
            
            # Check if swapping maintains disjointness in both layers
            # Remove c1 from L1, add to L2. Remove c2 from L2, add to L1.
            # Check L1 - c1 + c2 and L2 - c2 + c1
            
            wires1 = set()
            for c in current_layers[l1]:
                if c != c1:
                    wires1.add(c[0])
                    wires1.add(c[1])
            if c2[0] in wires1 or c2[1] in wires1:
                consecutive_failures += 1
                continue
                
            wires2 = set()
            for c in current_layers[l2]:
                if c != c2:
                    wires2.add(c[0])
                    wires2.add(c[1])
            if c1[0] in wires2 or c1[1] in wires2:
                consecutive_failures += 1
                continue
                
            candidate_layers = current_layers[:]
            # Remove c1 from l1
            candidate_layers[l1] = [c for c in candidate_layers[l1] if c != c1]
            candidate_layers[l1].append(c2)
            
            # Remove c2 from l2
            candidate_layers[l2] = [c for c in candidate_layers[l2] if c != c2]
            candidate_layers[l2].append(c1)
            
            flat = _flatten_layers(candidate_layers)
            if _verify_network_bit_parallel(flat):
                current_layers = candidate_layers
                consecutive_failures = 0
                if len(flat) < best_len:
                    best_flat = flat
                    best_layers = [list(l) for l in current_layers]
                    best_len = len(flat)
            else:
                consecutive_failures += 1
                
        else:
            # Modify a comparator target
            if len(current_flat) == 0:
                continue
            l_idx = random.randint(0, len(current_layers) - 1)
            if not current_layers[l_idx]:
                continue
            c_idx = random.randint(0, len(current_layers[l_idx]) - 1)
            i, j = current_layers[l_idx][c_idx]
            
            # Pick a new target for i or j
            if random.random() < 0.5:
                new_j = random.randint(i + 1, N - 1)
                new_comp = (i, new_j)
            else:
                new_i = random.randint(0, j - 1)
                new_comp = (new_i, j)
                
            # Check disjointness in layer
            wires_in_layer = set()
            valid = True
            for c in current_layers[l_idx]:
                if c != current_layers[l_idx][c_idx]:
                    if new_comp[0] == c[0] or new_comp[0] == c[1] or new_comp[1] == c[0] or new_comp[1] == c[1]:
                        valid = False
                        break
            if not valid:
                consecutive_failures += 1
                continue
                
            candidate_layers = current_layers[:]
            candidate_layers[l_idx] = candidate_layers[l_idx][:]
            candidate_layers[l_idx][c_idx] = new_comp
            
            flat = _flatten_layers(candidate_layers)
            if _verify_network_bit_parallel(flat):
                current_layers = candidate_layers
                consecutive_failures = 0
                if len(flat) < best_len:
                    best_flat = flat
                    best_layers = [list(l) for l in current_layers]
                    best_len = len(flat)
            else:
                consecutive_failures += 1

    if not _verify_network_bit_parallel(best_flat):
        best_flat = _flatten_layers(oet_comparators)
        
    return best_flat
