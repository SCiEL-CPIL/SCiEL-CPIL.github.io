"""Improved sorting network construction for 18 wires using a greedy set-cover heuristic."""
import time
import numpy as np

N = 18

# Precompute all 2^18 boolean inputs as a matrix of shape (2^18, 18)
# Each row is a bit-vector representing the input.
# We track which inputs are still "unsorted" (i.e., not in non-decreasing order).

def _build_inputs():
    """Return array of shape (2^N, N) with all boolean inputs."""
    # Use bit manipulation to generate all inputs efficiently
    n = N
    total = 1 << n
    # Create a uint8 array
    inputs = np.zeros((total, n), dtype=np.uint8)
    for bit in range(n):
        # For bit position 'bit', the pattern is: 0 for first 2^bit, 1 for next 2^bit, etc.
        # Actually, let's just use the binary representation
        # inputs[:, bit] = (np.arange(total) >> bit) & 1
        inputs[:, bit] = ((np.arange(total, dtype=np.uint32) >> bit) & 1).astype(np.uint8)
    return inputs

def _is_sorted_vec(arr):
    """Check if a 1D array is non-decreasing. Returns boolean."""
    return np.all(arr[:-1] <= arr[1:])

def _is_sorted_batch(inputs):
    """Check which rows in inputs (shape M x N) are sorted. Returns boolean array of length M."""
    # A row is sorted if inputs[i, j] <= inputs[i, j+1] for all j
    # Compare adjacent columns
    diffs = inputs[:, 1:] - inputs[:, :-1]  # shape (M, N-1)
    # Sorted if all diffs >= 0
    return np.all(diffs >= 0, axis=1)

def construct():
    """
    Construct a sorting network for N=18 wires using a greedy heuristic.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Precompute all boolean inputs
    inputs = _build_inputs()
    total_inputs = 1 << N
    
    # Track which inputs are still unsorted
    unsorted_mask = ~_is_sorted_batch(inputs)
    num_unsorted = np.sum(unsorted_mask)
    
    if num_unsorted == 0:
        return []
    
    # For each possible comparator (i, j) with i < j, we want to know
    # which unsorted inputs it would fix.
    # A comparator (i, j) swaps the values at positions i and j if inputs[i] > inputs[j].
    # After applying comparator (i, j), an input becomes sorted if it was sorted after the swap.
    
    # Strategy: Greedily select the comparator that fixes the most unsorted inputs.
    # To speed this up, we can precompute for each comparator which inputs it affects.
    
    # List of all possible comparators
    all_comparators = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # We'll maintain the current state of all inputs
    # But applying a comparator to all 2^18 inputs is expensive.
    # Instead, let's think differently.
    
    # Alternative approach: Use the fact that a comparator (i,j) fixes an input
    # if and only if:
    #   1. The input has inputs[i] > inputs[j] (so the swap happens), AND
    #   2. After swapping, the result is sorted.
    
    # But checking "after swapping is sorted" for all inputs is still expensive.
    
    # Let's try a different greedy strategy:
    # At each step, for each candidate comparator (i,j), count how many unsorted inputs
    # would become sorted. Pick the one with the maximum count.
    
    # To make this feasible within time, we'll use a simplified heuristic:
    # Only consider comparators between adjacent wires, or wires that are "close".
    
    # Actually, let's try a more practical approach:
    # Use a known good network as a starting point and try to prune it.
    
    # Start with odd-even transposition network
    network = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Verify it sorts
    def _apply_network(inputs, network):
        """Apply the network to all inputs, return the sorted outputs."""
        result = inputs.copy()
        for (i, j) in network:
            # For each input, if result[i] > result[j], swap them
            mask = result[:, i] > result[:, j]
            # Swap
            temp = result[:, i].copy()
            result[:, i] = np.where(mask, result[:, j], result[:, i])
            result[:, j] = np.where(mask, temp, result[:, j])
        return result
    
    # Verify the initial network
    sorted_outputs = _apply_network(inputs, network)
    assert np.all(_is_sorted_batch(sorted_outputs)), "Initial network doesn't sort!"
    
    # Now try to remove comparators one by one
    # For each comparator, check if removing it still results in a valid sorting network
    removed = []
    for idx in range(len(network)):
        if time.time() > deadline:
            break
        
        # Try removing network[idx]
        candidate = network[:idx] + network[idx+1:]
        
        # Check if candidate still sorts
        sorted_outputs = _apply_network(inputs, candidate)
        if np.all(_is_sorted_batch(sorted_outputs)):
            network = candidate
            removed.append(idx)
    
    # Try again with a different order - remove from the end
    for idx in range(len(network) - 1, -1, -1):
        if time.time() > deadline:
            break
        
        candidate = network[:idx] + network[idx+1:]
        sorted_outputs = _apply_network(inputs, candidate)
        if np.all(_is_sorted_batch(sorted_outputs)):
            network = candidate
    
    # Try swapping adjacent comparators to see if we can enable more removals
    for _ in range(3):  # A few passes
        if time.time() > deadline:
            break
        improved = True
        while improved:
            improved = False
            for idx in range(len(network) - 1):
                if time.time() > deadline:
                    break
                
                # Try swapping network[idx] and network[idx+1]
                candidate = network[:idx] + [network[idx+1], network[idx]] + network[idx+2:]
                
                # Check if candidate still sorts
                sorted_outputs = _apply_network(inputs, candidate)
                if np.all(_is_sorted_batch(sorted_outputs)):
                    network = candidate
                    improved = True
                    
                    # After swapping, try to remove again
                    for jdx in range(len(network)):
                        if time.time() > deadline:
                            break
                        cand2 = network[:jdx] + network[jdx+1:]
                        sorted_outputs2 = _apply_network(inputs, cand2)
                        if np.all(_is_sorted_batch(sorted_outputs2)):
                            network = cand2
                            improved = True
                            break
    
    return network

# Pre-compute at module load to save time during construct()
# But we need to be careful about memory and time
_inputs_cache = None

def _get_inputs():
    global _inputs_cache
    if _inputs_cache is None:
        _inputs_cache = _build_inputs()
    return _inputs_cache
