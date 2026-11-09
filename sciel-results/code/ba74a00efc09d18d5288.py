import numpy as np
import time

N = 16

def _generate_batcher_oes():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a list of (i, j) comparator pairs.
    """
    comparators = []
    
    def oes_merge(n, wires):
        """
        Recursively merges two sorted sequences of length n/2 located in 'wires'.
        'wires' is a list of global wire indices in the order they appear in the merge step.
        """
        if n == 1:
            return
        half = n // 2
        
        # Odd wires: wires[1], wires[3], ...
        # Even wires: wires[0], wires[2], ...
        odd_wires = wires[1::2]
        even_wires = wires[0::2]
        
        # Recursively merge the odd and even subsequences
        oes_merge(half, odd_wires)
        oes_merge(half, even_wires)
        
        # Final compare-exchanges between adjacent pairs
        for i in range(half):
            comparators.append((wires[2*i], wires[2*i+1]))

    def oes_sort(n, wires):
        """
        Recursively sorts the sequence in 'wires' using Odd-Even Merge Sort.
        """
        if n == 1:
            return
        half = n // 2
        left_wires = wires[:half]
        right_wires = wires[half:]
        
        # Sort left and right halves
        oes_sort(half, left_wires)
        oes_sort(half, right_wires)
        
        # Merge the two sorted halves
        oes_merge(n, wires)

    oes_sort(N, list(range(N)))
    return comparators

def _is_valid_skip(network, skip_idx, inputs):
    """
    Checks if the network (with comparator at skip_idx removed) sorts all inputs.
    """
    state = inputs.copy()
    for i, (a, b) in enumerate(network):
        if i == skip_idx:
            continue
        min_val = np.minimum(state[:, a], state[:, b])
        max_val = np.maximum(state[:, a], state[:, b])
        state[:, a] = min_val
        state[:, b] = max_val
    
    # Check if sorted
    diffs = np.diff(state, axis=1)
    return np.all(diffs >= 0)

def construct():
    start_time = time.time()
    # Use 50s to be safe against the 60s limit
    deadline = start_time + 50.0
    
    # Initial network: Batcher OES
    network = _generate_batcher_oes()
    
    # Pre-compute all 2^16 binary inputs
    # bits[:, i] is the i-th bit of the row index
    bits = np.arange(2**N)
    inputs = ((bits[:, None] >> np.arange(N)) & 1).astype(np.int8)
    
    # Local Search: Greedy Removal
    # We try to remove comparators one by one.
    # We iterate in reverse order to potentially remove later comparators first,
    # which might be more redundant in some structures.
    
    while time.time() < deadline:
        removed_any = False
        # Iterate in reverse
        for idx in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Check if removing network[idx] keeps the network valid
            if _is_valid_skip(network, idx, inputs):
                network.pop(idx)
                removed_any = True
        
        if not removed_any:
            # No more comparators can be removed in this pass
            # We could try a different order (e.g. forward) to escape local minima?
            # But given time constraints, we assume convergence.
            # Or we can break to save time if we've reached a good size.
            # Let's do one more forward pass just in case.
            forward_removed = False
            for idx in range(len(network)):
                if time.time() > deadline:
                    break
                if _is_valid_skip(network, idx, inputs):
                    network.pop(idx)
                    forward_removed = True
            
            if not forward_removed:
                break
    return network
