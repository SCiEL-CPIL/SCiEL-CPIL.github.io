"""Size-efficient sorting networks on 16 wires."""
import time
import numpy as np

N = 16
DEADLINE = 90.0
SAFETY_MARGIN = 5.0  # seconds

# EVOLVE-BLOCK-START
def _apply_comparators_fast(inputs, comparators):
    """Apply a list of comparators to a set of inputs (numpy array of shape (num_inputs, N))."""
    x = inputs.copy()
    for i, j in comparators:
        min_vals = np.minimum(x[:, i], x[:, j])
        max_vals = np.maximum(x[:, i], x[:, j])
        x[:, i] = min_vals
        x[:, j] = max_vals
    return x

def _is_sorted_network_fast(comparators):
    """Check if the network sorts all 2^N boolean inputs using zero-one principle."""
    n_inputs = 1 << N
    bits = np.arange(n_inputs, dtype=np.uint16)
    inputs = ((bits[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    
    result = _apply_comparators_fast(inputs, comparators)
    
    # Check if all rows are sorted (non-decreasing)
    diffs = result[:, 1:] - result[:, :-1]
    return np.all(diffs >= 0)

def construct():
    """Construct a sorting network for 16 wires with as few comparators as possible."""
    start_time = time.time()
    
    # Start with odd-even transposition (120 comparators)
    best_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    # Verify the starting network
    if not _is_sorted_network_fast(best_network):
        raise RuntimeError("Initial network is not valid")
    
    # Greedy removal: try removing each comparator
    improved = True
    while improved and (time.time() - start_time) < (DEADLINE - SAFETY_MARGIN):
        improved = False
        for idx in range(len(best_network) - 1, -1, -1):
            if (time.time() - start_time) >= (DEADLINE - SAFETY_MARGIN):
                break
            
            candidate = best_network[:idx] + best_network[idx+1:]
            
            if _is_sorted_network_fast(candidate):
                best_network = candidate
                improved = True
                break
    
    return best_network
# EVOLVE-BLOCK-END
