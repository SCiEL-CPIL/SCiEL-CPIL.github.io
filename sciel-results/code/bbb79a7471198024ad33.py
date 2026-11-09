import time
import numpy as np

N = 17
NUM_BOOL_INPUTS = 1 << N  # 131072
DEADLINE_SEC = 85.0  # Safety margin under 90s wall-clock


def _make_full_bool_inputs():
    """Generate all 2^N boolean inputs as a uint8 array of shape (2^N, N)."""
    arr = np.zeros((NUM_BOOL_INPUTS, N), dtype=np.uint8)
    for bit in range(N):
        block = 2 ** bit
        num_blocks = NUM_BOOL_INPUTS // (2 * block)
        one_block = np.zeros(2 * block, dtype=np.uint8)
        one_block[block:] = 1
        arr[:, bit] = np.tile(one_block, num_blocks)
    return arr


def _simulate_network(network, inputs):
    """
    Simulate the network on all 2^N boolean inputs.
    Returns True if sorted.
    """
    state = inputs.copy()
    for (i, j) in network:
        a = state[:, i]
        b = state[:, j]
        # min to i, max to j
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    # Check sortedness
    # For each adjacent pair (k, k+1), check if state[:, k] <= state[:, k+1]
    # We can do this in one vectorized check for all pairs
    # state[:, :-1] <= state[:, 1:]
    if np.any(state[:, :-1] > state[:, 1:]):
        return False
    return True


def _odd_even_transposition_network():
    """Standard odd-even transposition sorting network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net


def construct():
    start_time = time.time()
    
    # 1. Generate boolean inputs
    bool_inputs = _make_full_bool_inputs()
    
    # 2. Initialize with a known valid network
    network = _odd_even_transposition_network()
    
    # Verify initial network (should be true)
    if not _simulate_network(network, bool_inputs):
        # Fallback if something went wrong, though it shouldn't
        return network

    # 3. Greedy Deletion Optimization
    # We repeatedly scan the network and remove any comparator whose removal
    # preserves the sorting property. We restart the scan after each removal.
    
    improved = True
    while improved:
        improved = False
        # Iterate backwards to handle index shifts if we remove, 
        # or just break and restart. Breaking and restarting is safer and simpler.
        for idx in range(len(network)):
            if time.time() > DEADLINE_SEC:
                return network
            
            # Try removing network[idx]
            candidate = network[:idx] + network[idx+1:]
            
            # Check validity
            if _simulate_network(candidate, bool_inputs):
                # Removal is valid, update network
                network = candidate
                improved = True
                break # Restart the loop with the new, shorter network
    
    return network
