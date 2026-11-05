"""Size-efficient sorting networks on 17 wires - optimized local search."""
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
    """Check if all rows are sorted (non-decreasing)."""
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    """Check if network sorts all boolean inputs."""
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _build_checkpoints(network, all_inputs):
    """Build checkpoint states: state before each comparator.
    Returns list of (i, j, state_before) tuples."""
    state = all_inputs.copy()
    checkpoints = []
    for idx, (i, j) in enumerate(network):
        checkpoints.append((i, j, state.copy()))
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return checkpoints

def _test_removal(checkpoints, idx, all_inputs):
    """Test if removing comparator at idx still sorts, using checkpoint at idx.
    Apply remaining comparators from idx+1 onward to checkpoint state."""
    if idx >= len(checkpoints):
        return False
    i, j, state_before = checkpoints[idx]
    state = state_before
    for k in range(idx + 1, len(checkpoints)):
        ik, jk, _ = checkpoints[k]
        min_val = np.minimum(state[:, ik], state[:, jk])
        max_val = np.maximum(state[:, ik], state[:, jk])
        state[:, ik] = min_val
        state[:, jk] = max_val
    return _is_sorted_fast(state)

def _test_two_removals(checkpoints, idx1, idx2, all_inputs):
    """Test if removing both comparators at idx1 and idx2 still sorts."""
    if idx1 >= len(checkpoints) or idx2 >= len(checkpoints):
        return False
    # Start from the earlier checkpoint
    start = min(idx1, idx2)
    i, j, state_before = checkpoints[start]
    state = state_before
    for k in range(start + 1, len(checkpoints)):
        if k == idx1 or k == idx2:
            continue
        ik, jk, _ = checkpoints[k]
        min_val = np.minimum(state[:, ik], state[:, jk])
        max_val = np.maximum(state[:, ik], state[:, jk])
        state[:, ik] = min_val
        state[:, jk] = max_val
    return _is_sorted_fast(state)

def _local_search_aggressive(network, all_inputs, deadline):
    """Aggressive local search using checkpoint-based removal testing."""
    network = list(network)
    
    for _pass in range(20):
        if time.time() > deadline:
            break
        improved = False
        n = len(network)
        
        # Build checkpoints for current network
        checkpoints = _build_checkpoints(network, all_inputs)
        
        # Try removals from the end (more likely redundant)
        for i in range(n - 1, -1, -1):
            if time.time() > deadline:
                return network
            if _test_removal(checkpoints, i, all_inputs):
                network = network[:i] + network[i+1:]
                improved = True
                break
        
        if not improved:
            # Try from the beginning
            for i in range(n):
                if time.time() > deadline:
                    return network
                if _test_removal(checkpoints, i, all_inputs):
                    network = network[:i] + network[i+1:]
                    improved = True
                    break
        
        if not improved:
            break
    
    return network

def _two_removal_search(network, all_inputs, deadline):
    """Try removing pairs of comparators to escape local minima."""
    network = list(network)
    n = len(network)
    
    if n < 2:
        return network
    
    checkpoints = _build_checkpoints(network, all_inputs)
    
    # Try pairs of adjacent or near-adjacent comparators
    for i in range(n - 1):
        if time.time() > deadline:
            break
        for j in range(i + 1, min(i + 4, n)):
            if time.time() > deadline:
                break
            if _test_two_removals(checkpoints, i, j, all_inputs):
                # Remove both, higher index first to preserve indices
                if j > i:
                    network = network[:i] + network[i+1:j] + network[j+1:]
                else:
                    network = network[:j] + network[j+1:i] + network[i+1:]
                return network
    
    return network

def _try_swap_and_remove(network, i, all_inputs):
    """Try swapping network[i] and network[i+1], then removing one of them."""
    if i >= len(network) - 1:
        return None
    if network[i] == network[i+1]:
        return None
    
    new_network = network.copy()
    new_network[i], new_network[i+1] = new_network[i+1], new_network[i]
    
    # Try removing i
    test1 = new_network[:i] + new_network[i+1:]
    final1 = _simulate_network(test1, all_inputs)
    if _is_sorted_fast(final1):
        return test1
    
    # Try removing i+1
    test2 = new_network[:i+1] + new_network[i+2:]
    final2 = _simulate_network(test2, all_inputs)
    if _is_sorted_fast(final2):
        return test2
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin within 90s limit
    
    # Start with odd-even transposition
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Phase 1: Aggressive single-removal local search with checkpoints
    network = _local_search_aggressive(network, all_inputs, deadline)
    
    # Phase 2: Two-removal search to escape local minima
    if time.time() < deadline - 10.0:
        network = _two_removal_search(network, all_inputs, deadline - 10.0)
    
    # Phase 3: Swap-and-remove to enable further reductions
    if time.time() < deadline - 8.0:
        improved = True
        while improved and time.time() < deadline - 8.0:
            improved = False
            for i in range(len(network) - 1):
                if time.time() > deadline - 8.0:
                    break
                result = _try_swap_and_remove(network, i, all_inputs)
                if result is not None:
                    network = result
                    improved = True
                    break
    
    # Phase 4: Final single-removal pass after swaps
    if time.time() < deadline - 3.0:
        network = _local_search_aggressive(network, all_inputs, deadline - 3.0)
    
    # Phase 5: Another two-removal attempt
    if time.time() < deadline - 2.0:
        network = _two_removal_search(network, all_inputs, deadline - 2.0)
    
    return network
