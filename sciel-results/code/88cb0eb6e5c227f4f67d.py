"""Size-efficient sorting networks on 17 wires."""
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

def _find_redundant_comparators(network, all_inputs):
    """Find all redundant comparators in one pass."""
    n_comps = len(network)
    if n_comps == 0:
        return set()
    
    # Simulate and record the two relevant wire values before each comparator
    state = all_inputs.copy()
    wire_vals_before = []
    for i, j in network:
        wire_vals_before.append((state[:, i].copy(), state[:, j].copy()))
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    # Check which comparators are redundant
    redundant = set()
    for k, (vi, vj) in enumerate(wire_vals_before):
        if not np.any(vi > vj):
            redundant.add(k)
    
    return redundant

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively remove redundant comparators."""
    network = list(network)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        
        redundant = _find_redundant_comparators(network, all_inputs)
        if not redundant:
            break
        
        # Remove all redundant comparators at once
        network = [c for i, c in enumerate(network) if i not in redundant]
        improved = True
    
    return network

def construct():
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin within 90s limit
    
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
