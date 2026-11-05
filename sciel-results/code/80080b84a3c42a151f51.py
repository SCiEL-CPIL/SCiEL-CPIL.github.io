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

def _simulate_network_early_exit(network, inputs, skip_idx=None, skip_idx2=None):
    """Simulate, skipping one or two comparators; return None if found unsorted early."""
    state = inputs.copy()
    for k, (i, j) in enumerate(network):
        if k == skip_idx or k == skip_idx2:
            continue
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    # final check
    diffs = np.diff(state, axis=1)
    if np.any(diffs < 0):
        return None
    return state

def _is_sorted_fast(state):
    diffs = np.diff(state, axis=1)
    return not np.any(diffs < 0)

def _is_sorting_network_fast(network, all_inputs):
    final = _simulate_network(network, all_inputs)
    return _is_sorted_fast(final)

def _try_remove_comparator(network, idx, all_inputs):
    """Try removing comparator at index idx. Returns True if network still sorts."""
    return _simulate_network_early_exit(network, all_inputs, skip_idx=idx) is not None

def _try_remove_two_comparators(network, idx, jdx, all_inputs):
    """Try removing two comparators. Returns True if network still sorts."""
    if idx == jdx:
        return False
    return _simulate_network_early_exit(network, all_inputs, skip_idx=idx, skip_idx2=jdx) is not None

def _local_search_remove(network, all_inputs, deadline):
    """Iteratively try to remove comparators, prioritizing later ones, with pair removals."""
    network = list(network)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Single removals, end-first
        for i in range(len(network) - 1, -1, -1):
            if time.time() > deadline:
                return network
            if _try_remove_comparator(network, i, all_inputs):
                network.pop(i)
                changed = True
                break
        if not changed:
            # Try pair removals: for each i, try removing i with a nearby j
            for i in range(len(network) - 1, -1, -1):
                if time.time() > deadline:
                    return network
                # try pairing with a few other indices
                for j in range(i - 1, -1, -1):
                    if time.time() > deadline:
                        return network
                    if _try_remove_two_comparators(network, i, j, all_inputs):
                        # remove higher index first
                        if i > j:
                            network.pop(i)
                            network.pop(j)
                        else:
                            network.pop(j)
                            network.pop(i)
                        changed = True
                        break
                if changed:
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

    # Local search: try to remove comparators (single + pairs)
    network = _local_search_remove(network, all_inputs, deadline)

    # Additional optimization: try to swap adjacent comparators to enable further removals
    if time.time() < deadline - 5.0:
        improved = True
        while improved and time.time() < deadline - 5.0:
            improved = False
            for i in range(len(network) - 1):
                if time.time() > deadline - 5.0:
                    break
                # Try swapping network[i] and network[i+1]
                if network[i] != network[i+1]:
                    new_network = network.copy()
                    new_network[i], new_network[i+1] = new_network[i+1], new_network[i]
                    if _is_sorting_network_fast(new_network, all_inputs):
                        # Check if this swap enables removal of one of them
                        for remove_idx in [i, i+1]:
                            test_network = new_network[:remove_idx] + new_network[remove_idx+1:]
                            if _is_sorting_network_fast(test_network, all_inputs):
                                network = test_network
                                improved = True
                                break
                        if improved:
                            break

    return network
# EVOLVE-BLOCK-END
