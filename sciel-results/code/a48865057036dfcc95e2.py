import numpy as np
import time

N = 16

# EVOLVE-BLOCK-START
def _bitonic_merge_network(n):
    """Generate comparators for bitonic merge of size n."""
    comps = []
    k = n // 2
    while k >= 1:
        for i in range(n - k):
            comps.append((i, i + k))
        k //= 2
    return comps

def _bitonic_sort_network(n):
    """Generate Batcher's Bitonic Sort network for n wires."""
    if n == 1:
        return []
    half = n // 2
    # Left half: ascending
    left = _bitonic_sort_network(half)
    
    # Right half: descending
    # Generate ascending network for half, then reverse indices
    right_asc = _bitonic_sort_network(half)
    right_desc = []
    for i, j in right_asc:
        # Reverse indices within the half-block to make it descending
        # Original (i, j) with i < j.
        # New indices: (half - 1 - j, half - 1 - i)
        # Since i < j, half-1-j < half-1-i.
        i_new = half - 1 - j
        j_new = half - 1 - i
        right_desc.append((half + i_new, half + j_new))
        
    # Merge step
    merge = _bitonic_merge_network(n)
    
    return left + right_desc + merge

def construct():
    # Generate initial Batcher network
    net = _bitonic_sort_network(N)
    
    # Precompute inputs for validity check
    # 2^16 = 65536 inputs, 16 wires
    # Shape: (16, 65536)
    inputs = np.array(np.unravel_index(np.arange(2**N), (2,)*N)).T
    
    def is_valid(network):
        wires = inputs.copy()
        for i, j in network:
            min_val = np.minimum(wires[i], wires[j])
            max_val = np.maximum(wires[i], wires[j])
            wires[i] = min_val
            wires[j] = max_val
        # Check if sorted: diff >= 0
        diffs = np.diff(wires, axis=0)
        return np.all(diffs >= 0)
    
    best_net = net
    best_count = len(net)
    
    start_time = time.time()
    timeout = 80.0 # Safety margin
    
    # Local Search: Try to remove comparators
    # We iterate through the network and try removing each comparator
    # If removal keeps the network valid, we accept it.
    # We repeat this process until no more removals are possible or time runs out.
    
    # To make it faster, we can randomize the order of removal attempts.
    # Also, we can try to remove multiple comparators if possible.
    
    while time.time() - start_time < timeout:
        improved = False
        # Create a list of indices to try
        indices = list(range(len(best_net)))
        np.random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > timeout:
                break
            
            # Skip if index is out of bounds (in case net changed)
            if idx >= len(best_net):
                continue
                
            # Try removing best_net[idx]
            candidate = best_net[:idx] + best_net[idx+1:]
            
            # Only check if it reduces the count
            if len(candidate) < best_count:
                if is_valid(candidate):
                    best_net = candidate
                    best_count = len(best_net)
                    improved = True
                    # Break and restart the loop with new indices
                    break
                    
        if not improved:
            # If no single removal worked, we might be at a local minimum.
            # We could try swapping, but for simplicity and speed, we break.
            # Given the time limit, breaking is safer.
            break
            
    return best_net
# EVOLVE-BLOCK-END
