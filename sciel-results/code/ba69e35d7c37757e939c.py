import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _generate_oe_sort_network(n):
    """Generate Odd-Even Transposition Sorting Network."""
    network = []
    for phase in range(n):
        if phase % 2 == 0:
            for i in range(0, n, 2):
                if i + 1 < n:
                    network.append((i, i + 1))
        else:
            for i in range(1, n, 2):
                if i + 1 < n:
                    network.append((i, i + 1))
    return network

def _verify_network(network, inputs):
    """Verify if network sorts all inputs. inputs is (M, N) numpy array."""
    state = inputs.copy()
    for i, j in network:
        min_ij = np.minimum(state[:, i], state[:, j])
        max_ij = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_ij
        state[:, j] = max_ij
    # Check if sorted: state[:, k] <= state[:, k+1]
    return np.all(state[:, :-1] <= state[:, 1:])

def construct():
    """Construct a sorting network for N wires."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # 1. Generate initial correct network (Odd-Even Transposition)
    network = _generate_oe_sort_network(N)
    
    # 2. Prepare all Boolean inputs for verification
    max_inputs = 1 << N
    idx = np.arange(max_inputs, dtype=np.uint32)
    # Generate bits
    # Ensure arange is uint32 to match idx
    shifts = np.arange(N, dtype=np.uint32)
    inputs = (idx[:, None] >> shifts) & 1
    inputs = inputs.astype(np.uint8)
    
    # 3. Greedy Removal
    # Try to remove comparators from the end of the network.
    i = len(network) - 1
    while i >= 0 and time.time() < deadline:
        # Tentatively remove network[i]
        # Instead of slicing and copying the whole list every time (O(N)),
        # we can just simulate with a flag or a modified list.
        # Slicing is O(N) where N~150. 150 * 1000 iterations is small.
        candidate_network = network[:i] + network[i+1:]
        
        # Verify
        if _verify_network(candidate_network, inputs):
            # Accept removal
            network = candidate_network
            # Keep i same to check the new element at position i
        else:
            # Cannot remove, move to previous index
            i -= 1
            
    return network

# EVOLVE-BLOCK-END
