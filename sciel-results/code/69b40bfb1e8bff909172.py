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
    """
    Find all comparators that are currently 'no-ops' (already sorted at that point).
    Returns a set of indices.
    """
    n_comps = len(network)
    if n_comps == 0:
        return set()
    
    state = all_inputs.copy()
    # We store the state of the two wires BEFORE the comparator for each step
    # This allows us to check if the comparator actually did anything.
    # If state[i] <= state[j] for ALL inputs before the comparator, it's redundant.
    
    # Optimization: We only need to check if any input has state[i] > state[j].
    # We can do this in a single pass if we store the 'violations' or just check on the fly.
    # However, we need to simulate the whole network to get the state before each step.
    
    # Let's simulate and record the "active" status.
    redundant = set()
    
    # To be efficient, we can check redundancy in batches or just one pass.
    # One pass:
    for k, (i, j) in enumerate(network):
        # Check if any input has state[i] > state[j]
        # If np.any(state[:, i] > state[:, j]) is False, then it's redundant.
        if not np.any(state[:, i] > state[:, j]):
            redundant.add(k)
        # Apply comparator
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    return redundant

def _local_search_remove(network, all_inputs, deadline):
    """
    Iteratively remove redundant comparators.
    Strategy:
    1. Find all currently redundant comparators.
    2. Remove them all at once.
    3. Repeat until no more redundant comparators are found or time runs out.
    
    This is much faster than trying one by one because removing one comparator
    might make another redundant, but removing all currently redundant ones
    is safe and reduces the network size significantly in each iteration.
    """
    network = list(network)
    
    # Initial check: is the network valid?
    if not _is_sorting_network_fast(network, all_inputs):
        return network
        
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            return network
        
        redundant = _find_redundant_comparators(network, all_inputs)
        if not redundant:
            break
        
        # Remove all redundant comparators
        # We keep the ones NOT in the redundant set
        network = [c for idx, c in enumerate(network) if idx not in redundant]
        improved = True
        
        # Optional: If we removed a lot, we might want to do a full verification
        # to ensure we didn't break something (though logically removing no-ops is safe).
        # But due to floating point or logic errors, a check is cheap insurance.
        # However, _find_redundant_comparators already simulated the whole thing.
        # If a comparator was a no-op in the full simulation, removing it changes nothing
        # for the subsequent steps. So the network remains valid.
        
    return network

def construct():
    start_time = time.time()
    # The evaluator kills after 90s. We should finish well before.
    # 50s deadline is a safe margin.
    deadline = start_time + 50.0
    
    # Start with odd-even transposition
    # For N wires, we need N phases.
    # Phase k (0-indexed): comparators at (0,1), (2,3)... if k even, (1,2), (3,4)... if k odd.
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
            
    # Generate all boolean inputs
    all_inputs = _all_boolean_inputs(N)
    
    # Verify initial network
    if not _is_sorting_network_fast(network, all_inputs):
        return network
    
    # Local search: try to remove comparators
    network = _local_search_remove(network, all_inputs, deadline)
    
    return network
