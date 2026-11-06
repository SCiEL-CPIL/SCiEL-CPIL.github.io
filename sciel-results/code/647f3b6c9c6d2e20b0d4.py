"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute masks for fast simulation
# A comparator (i, j) sets wire i to min(x_i, x_j) and wire j to max(x_i, x_j)
# For 17 wires, we can represent state as a 17-bit integer.
# However, simulating each comparator step-by-step on 131k inputs is slow in pure Python.
# We will use NumPy vectorization over all inputs at once for speed.

# Pre-compute the indices for each wire in a flattened array
# Inputs: shape (131072, 17)
# Comparators: list of (i, j)

def simulate_network(comparators, inputs):
    """
    inputs: numpy array of shape (num_inputs, N) with values in {0, 1}
    comparators: list of (i, j)
    returns: sorted numpy array of shape (num_inputs, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

# Precompute the input matrix and expected sorted outputs once at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 4096
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]

def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    
    # Simulate the network on the precomputed inputs
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check if state matches sorted inputs
    return np.array_equal(state, _SORTED_INPUTS)

def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs. Not sufficient for final validation,
    but much faster for guiding search.
    """
    if len(comparators) > 4096:
        return False
    
    state = _SAMPLE_INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    return np.array_equal(state, _SAMPLE_SORTED)

# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)

def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    left = odd_even_merge_sort_network(half)
    right = odd_even_merge_sort_network(n - half)
    
    # Offset right half comparators
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two halves
    merged = left + right
    # Odd-even merge: interleave comparators between the two halves
    for k in range(1, n, 2):
        if k < n:
            merged.append((k - 1, k))
    
    return merged

def odd_even_merge_sort_network_v2(n):
    """
    Better odd-even merge sort network construction.
    """
    comparators = []
    
    def build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = build(half)
        right = build(n - half)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge phase: odd-even merge
        merge_comps = []
        # We need to merge two sorted sequences of length half and n-half
        # Using the odd-even merge procedure
        for phase in range(n - 1):
            start = phase % 2
            for i in range(start, n - 1, 2):
                # Compare wire i and i+1, but only if they span the boundary
                # Actually, for odd-even merge, we compare specific pairs
                pass
        
        # Simpler: just use the standard merge network
        # For merging two sorted lists of sizes a and b, we need a+b-1 comparators
        # arranged in a specific pattern
        a = half
        b = n - half
        
        # The merge network for two sorted inputs of size a and b
        # uses a+b-1 comparators in a specific arrangement
        merge = []
        for i in range(a + b - 1):
            # This is a simplified version; real odd-even merge is more complex
            # Let's just use a simple approach
            pass
        
        return left + right + merge
    
    return build(n)

# Use a known-good small network construction
def bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires.
    n must be a power of 2 for the standard construction.
    For n=17, we'll pad to 32 or use a hybrid approach.
    Here we return the network for the largest power of 2 <= n,
    plus additional comparators for the remaining wires.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For non-power-of-2, we'll use a different approach
    # Let's just build it recursively with the bitonic pattern
    comparators = []
    
    def bitonic(k, direction=1):
        """Generate bitonic merge network for k wires."""
        if k == 1:
            return []
        if k == 2:
            if direction == 1:
                return [(0, 1)]
            else:
                return [(1, 0)]  # swap
        
        half = k // 2
        # Recursively sort halves in opposite directions
        left = bitonic(half, 1)
        right = bitonic(k - half, -1)
        right = [(i + half, j + half) for i, j in right]
        
        # Merge: compare pairs (i, i+half)
        merge = []
        for i in range(half):
            j = i + half
            if direction == 1:
                merge.append((i, j))
            else:
                merge.append((j, i))
        
        return left + right + merge
    
    # For n=17, we can't use pure bitonic (needs power of 2)
    # Instead, use a layered approach
    return []

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin: finish well before 90s limit
    
    # Build a valid seed network using odd-even transposition (136 comparators)
    seed_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            seed_network.append((i, i + 1))
    
    best_network = list(seed_network)
    best_len = len(best_network)
    
    # Convert to layer representation: each layer contains disjoint comparators
    def to_layers(comparators):
        """Convert flat list to layers of disjoint comparators."""
        layers = []
        for comp in comparators:
            # Find a layer where this comparator can be added (no wire conflict)
            placed = False
            for layer in layers:
                wires_in_layer = set()
                for c in layer:
                    wires_in_layer.add(c[0])
                    wires_in_layer.add(c[1])
                if comp[0] not in wires_in_layer and comp[1] not in wires_in_layer:
                    layer.append(comp)
                    placed = True
                    break
            if not placed:
                layers.append([comp])
        return layers
    
    def from_layers(layers):
        """Convert layers back to flat list."""
        comps = []
        for layer in layers:
            comps.extend(layer)
        return comps
    
    # Layer-based simulation: apply all comparators in a layer simultaneously
    def simulate_layers(layers, inputs):
        """Simulate network given as layers on input matrix."""
        state = inputs.copy()
        for layer in layers:
            for i, j in layer:
                min_val = np.minimum(state[:, i], state[:, j])
                max_val = np.maximum(state[:, i], state[:, j])
                state[:, i] = min_val
                state[:, j] = max_val
        return state
    
    def is_sorted_layers(layers):
        """Full validation of layer-based network."""
        state = _INPUTS.copy()
        for layer in layers:
            for i, j in layer:
                min_val = np.minimum(state[:, i], state[:, j])
                max_val = np.maximum(state[:, i], state[:, j])
                state[:, i] = min_val
                state[:, j] = max_val
        return np.array_equal(state, _SORTED_INPUTS)
    
    def is_sorted_layers_sample(layers):
        """Sample validation of layer-based network."""
        state = _SAMPLE_INPUTS.copy()
        for layer in layers:
            for i, j in layer:
                min_val = np.minimum(state[:, i], state[:, j])
                max_val = np.maximum(state[:, i], state[:, j])
                state[:, i] = min_val
                state[:, j] = max_val
        return np.array_equal(state, _SAMPLE_SORTED)
    
    def count_comparators(layers):
        return sum(len(layer) for layer in layers)
    
    # Start with layered representation
    current_layers = to_layers(seed_network)
    
    # Phase 1: Try removing comparators from layers
    # For each comparator, check if removing it still gives a valid network
    i = 0
    while i < count_comparators(current_layers) and time.time() < deadline - 15.0:
        # Find the i-th comparator
        flat = from_layers(current_layers)
        if i >= len(flat):
            break
        comp = flat[i]
        
        # Create candidate with this comparator removed
        candidate_layers = []
        for layer in current_layers:
            new_layer = [c for c in layer if c != comp]
            if new_layer:
                candidate_layers.append(new_layer)
        
        # Validate
        if is_sorted_layers_sample(candidate_layers):
            if is_sorted_layers(candidate_layers):
                current_layers = candidate_layers
                best_network = from_layers(current_layers)
                best_len = len(best_network)
                continue  # Don't increment i, check same position again
        i += 1
    
    # Phase 2: Multiple passes of systematic deletion
    for _ in range(10):
        if time.time() > deadline - 12.0:
            break
        progress = False
        i = 0
        while i < count_comparators(current_layers) and time.time() < deadline - 12.0:
            flat = from_layers(current_layers)
            if i >= len(flat):
                break
            comp = flat[i]
            
            candidate_layers = []
            for layer in current_layers:
                new_layer = [c for c in layer if c != comp]
                if new_layer:
                    candidate_layers.append(new_layer)
            
            if is_sorted_layers_sample(candidate_layers):
                if is_sorted_layers(candidate_layers):
                    current_layers = candidate_layers
                    best_network = from_layers(current_layers)
                    best_len = len(best_network)
                    progress = True
                    continue
            i += 1
        if not progress:
            break
    
    # Phase 3: Try merging layers or restructuring
    # If two adjacent layers have no wire conflicts, we can merge them
    if time.time() < deadline - 8.0:
        for _ in range(50):
            if time.time() > deadline - 8.0:
                break
            if len(current_layers) < 2:
                break
            
            # Try removing a comparator from a random position
            flat = from_layers(current_layers)
            if not flat:
                break
            idx = random.randint(0, len(flat) - 1)
            comp = flat[idx]
            
            candidate_layers = []
            for layer in current_layers:
                new_layer = [c for c in layer if c != comp]
                if new_layer:
                    candidate_layers.append(new_layer)
            
            if is_sorted_layers_sample(candidate_layers):
                if is_sorted_layers(candidate_layers):
                    current_layers = candidate_layers
                    best_network = from_layers(current_layers)
                    best_len = len(best_network)
    
    # Phase 4: Try replacing comparators with different ones
    if time.time() < deadline - 5.0:
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            flat = from_layers(current_layers)
            if not flat:
                break
            idx = random.randint(0, len(flat) - 1)
            old_comp = flat[idx]
            
            # Try a random replacement
            i_new = random.randint(0, N - 2)
            j_new = random.randint(i_new + 1, N - 1)
            new_comp = (i_new, j_new)
            
            if new_comp == old_comp:
                continue
            
            # Build candidate with replacement
            candidate_layers = []
            for layer in current_layers:
                new_layer = []
                for c in layer:
                    if c == old_comp:
                        new_layer.append(new_comp)
                    else:
                        new_layer.append(c)
                if new_layer:
                    candidate_layers.append(new_layer)
            
            # Check if replacement creates wire conflicts within its layer
            valid = True
            for layer in candidate_layers:
                wires = set()
                for c in layer:
                    if c[0] in wires or c[1] in wires:
                        valid = False
                        break
                    wires.add(c[0])
                    wires.add(c[1])
                if not valid:
                    break
            
            if valid and is_sorted_layers_sample(candidate_layers):
                if is_sorted_layers(candidate_layers):
                    current_layers = candidate_layers
                    best_network = from_layers(current_layers)
                    best_len = len(best_network)
    
    # Phase 5: Final systematic pass
    i = 0
    while i < count_comparators(current_layers) and time.time() < deadline - 2.0:
        flat = from_layers(current_layers)
        if i >= len(flat):
            break
        comp = flat[i]
        
        candidate_layers = []
        for layer in current_layers:
            new_layer = [c for c in layer if c != comp]
            if new_layer:
                candidate_layers.append(new_layer)
        
        if is_sorted_layers_sample(candidate_layers):
            if is_sorted_layers(candidate_layers):
                current_layers = candidate_layers
                best_network = from_layers(current_layers)
                best_len = len(best_network)
                continue
        i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network
        
    return best_network
# EVOLVE-BLOCK-END
