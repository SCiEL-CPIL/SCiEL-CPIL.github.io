"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 17
import time
import numpy as np

def _all_boolean_inputs(n):
    """Generate all 2^n boolean inputs as a 2D array of shape (2^n, n)."""
    total = 1 << n
    indices = np.arange(total, dtype=np.int32)[:, None]
    bit_positions = np.arange(n, dtype=np.int32)[None, :]
    return ((indices >> bit_positions) & 1).astype(np.int8)

def _simulate_network(network, inputs):
    """Simulate a compare-exchange network on all inputs."""
    state = inputs.copy()
    for i, j in network:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state

def _is_sorted_fast(state):
    """Check if all rows are sorted (non-decreasing) using vectorized ops."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    new_network = network[:idx] + network[idx+1:]
    return _is_sorting_network_fast(new_network, all_inputs)

def _batch_try_removals(network, candidates, all_inputs):
    """Try removing each candidate index. Returns list of indices that can be removed."""
    removable = []
    for idx in candidates:
        if idx >= len(network):
            continue
        new_network = network[:idx] + network[idx+1:]
        final = _simulate_network(new_network, all_inputs)
        if _is_sorted_fast(final):
            removable.append(idx)
    return removable

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators, prioritizing later comparators."""
    network = list(network)
    improved = True
    while improved:
        improved = False
        # Try from the end first (later comparators are more likely redundant)
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break  # Restart after a removal
    # Second pass: try from the beginning
    improved = True
    while improved:
        improved = False
        for i in range(len(network)):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                improved = True
                break
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin within 90s limit
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    return network
# EVOLVE-BLOCK-END
