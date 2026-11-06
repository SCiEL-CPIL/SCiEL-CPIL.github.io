import time
import numpy as np

N = 18
TOTAL_INPUTS = 1 << N

def _precompute_inputs():
    """
    Precompute all 2^N Boolean inputs as a (N, 2^N) uint8 array.
    Column k represents the binary representation of k, with bit 0 as the first wire.
    """
    all_inputs = np.zeros((N, TOTAL_INPUTS), dtype=np.uint8)
    for i in range(N):
        # Bit i is 1 for every 2^(i+1) numbers, for the last 2^i of them
        # Pattern: 0...0 1...1 0...0 1...1
        # Period is 2^(i+1), first half 0, second half 1
        period = 1 << (i + 1)
        half = 1 << i
        pattern = np.zeros(period, dtype=np.uint8)
        pattern[half:] = 1
        all_inputs[i] = np.tile(pattern, TOTAL_INPUTS // period)
    return all_inputs

# Precompute at module level to save time in construct
_ALL_INPUTS = _precompute_inputs()

def _verify_network(network):
    """
    Vectorized verification of the sorting network.
    Returns True if the network sorts all 0-1 inputs.
    """
    # Copy the inputs to avoid modifying the global array
    x = _ALL_INPUTS.copy()
    
    for i, j in network:
        # Compare-exchange: min to i, max to j
        # We need to swap if x[i] > x[j]
        # Since values are 0 or 1, x[i] > x[j] is equivalent to x[i] == 1 and x[j] == 0
        # We can compute a mask of where a swap is needed
        mask = (x[i] == 1) & (x[j] == 0)
        if np.any(mask):
            # Use vectorized operations to update
            # x[i] becomes 0 where mask is true
            # x[j] becomes 1 where mask is true
            x[i][mask] = 0
            x[j][mask] = 1
            
    # Check if the result is sorted for all columns
    # A column is sorted if it is non-decreasing.
    # For binary values, this means there is no index k such that x[k] == 1 and x[k+1] == 0.
    # We can check this by computing the difference between adjacent rows.
    # If any element in x[1:] - x[:-1] is negative (i.e., 0 - 1 = -1), it's unsorted.
    # Alternatively, simply check if any (1, 0) pair exists in adjacent wires.
    diffs = x[1:].astype(np.int8) - x[:-1].astype(np.int8)
    # If any diff is -1, it means we have a 1 followed by a 0, which is unsorted.
    if np.any(diffs == -1):
        return False
    return True

def construct():
    start_time = time.time()
    # Safety margin: The evaluator kills at 90s, but the task says 60s. 
    # Let's use 55s to be safe for the 60s requirement mentioned in the text, 
    # or 85s if it's 90s. The prompt says "Keep an EXPLICIT deadline guard... 
    # The evaluator KILLS the program after 90s wall-clock." 
    # But the task description says "Each evaluation must finish within 60 s."
    # Let's target 55s to be safe for the 60s constraint, which is stricter.
    deadline = start_time + 55.0
    
    # Start with odd-even transposition network
    # Phases: N phases. Phase p starts at p%2.
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Verify initial network
    assert _verify_network(network), "Initial network is invalid"
    
    # Local search: Iterative removal
    # We repeatedly scan the network and remove any comparator that can be removed
    # without breaking the sorting property.
    # We scan from the end to the beginning to avoid index shifting issues 
    # (though we rebuild the list anyway).
    
    improved = True
    while improved and (time.time() - start_time) < deadline - 5:
        improved = False
        # Scan backwards
        for idx in range(len(network) - 1, -1, -1):
            if (time.time() - start_time) > deadline - 5:
                break
            
            # Try removing comparator at idx
            test_network = network[:idx] + network[idx+1:]
            
            if _verify_network(test_network):
                network = test_network
                improved = True
                
    # Second phase: Add-Remove cycles
    # Try adding a comparator, then trying to remove a different one to see if we can reduce size.
    # This helps escape local minima where no single removal is possible, but a swap is.
    
    existing_pairs = set(network)
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    max_rounds = 50
    for _round in range(max_rounds):
        if (time.time() - start_time) > deadline - 10:
            break
            
        found_improvement = False
        # Iterate through potential additions
        # To make it faster, we might shuffle or prioritize, but for now iterate all
        # We can break early if we find an improvement
        for add_pair in all_pairs:
            if (time.time() - start_time) > deadline - 10:
                break
                
            if add_pair in existing_pairs:
                continue
                
            # Add the new comparator
            test_network = network + [add_pair]
            
            # Now try to remove any comparator from this extended network
            # We want to end up with a network smaller than the current one
            # So we must remove at least one existing comparator (not the one we added, 
            # or if we remove the one we added, size is same, no improvement in size, 
            # but might enable further removals? The prompt asks for minimizing comparator count.
            # So we strictly want len < current len).
            
            # To be efficient, we can first check if the extended network is valid (it should be, 
            # as adding comparators preserves sorting).
            # Then try removing each comparator.
            
            # Optimization: Only try removing comparators that are "redundant" in some heuristic way?
            # For now, just try all.
            
            for idx in range(len(test_network) - 1, -1, -1):
                if (time.time() - start_time) > deadline - 10:
                    break
                    
                # Skip the one we just added? If we remove the one we just added, 
                # we get back to the original network, which is not an improvement in size.
                # So we must remove an index < len(test_network) - 1.
                if idx == len(test_network) - 1:
                    continue
                    
                removed_network = test_network[:idx] + test_network[idx+1:]
                
                if len(removed_network) < len(network):
                    if _verify_network(removed_network):
                        network = removed_network
                        existing_pairs = set(network)
                        found_improvement = True
                        break # Break the inner removal loop
                        
            if found_improvement:
                break # Break the addition loop
            
        if not found_improvement:
            # No improvement found in this round of add-remove cycles
            # We could try a different strategy, but for now, stop if no improvement
            # to save time, or continue a few more rounds?
            # Let's stop if no improvement to be safe with time.
            break
            
    return network

# Keep the dummy functions if they are expected by some interface, though construct is the main one
def _apply_network(network, x):
    x = list(x)
    for i, j in network:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    return x

def _is_sorted_network(network):
    for mask in range(1 << N):
        x = [(mask >> i) & 1 for i in range(N)]
        result = _apply_network(network, x)
        if result != sorted(result):
            return False
    return True

def _depth(network):
    return len(network)
