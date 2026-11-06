"""Size-efficient sorting networks on 16 wires - branch and bound search."""
import time
import numpy as np

N = 16
TOTAL_BOOLEANS = 1 << N  # 65536
DEADLINE = time.time() + 80  # 80s wall-clock safety margin

def _apply_network(network, inputs):
    """Apply a network to a batch of Boolean inputs (each row is a wire state).
    inputs: shape (num_inputs, N) uint8 array
    Returns: sorted array (num_inputs, N)
    """
    for (i, j) in network:
        # For each input, compare wire i and wire j
        a = inputs[:, i].copy()
        b = inputs[:, j].copy()
        # min goes to i, max goes to j
        inputs[:, i] = np.minimum(a, b)
        inputs[:, j] = np.maximum(a, b)
    return inputs

def _is_sorting_network(network):
    """Check if network sorts all 65536 Boolean inputs using zero-one principle."""
    # Generate all 65536 Boolean inputs
    # Each input is a 16-bit integer; we need to check if output is sorted
    # Use vectorized approach: generate all inputs as a (65536, 16) array
    
    # Create all possible 16-bit patterns
    all_patterns = np.arange(TOTAL_BOOLEANS, dtype=np.uint32)
    # Extract bits: for wire k, bit k of the pattern
    # We want inputs[:, k] = bit k of the pattern
    bits = np.zeros((TOTAL_BOOLEANS, N), dtype=np.uint8)
    for k in range(N):
        bits[:, k] = (all_patterns >> k) & 1
    
    # Apply network
    result = _apply_network(network, bits)
    
    # Check if each row is sorted: result[:, 0] <= result[:, 1] <= ... <= result[:, N-1]
    # For Boolean values, sorted means all 0s come before all 1s
    # So for each row, once we see a 1, all subsequent must be 1
    # Equivalently: the row should be of form [0,0,...,0,1,1,...,1]
    
    # Check: for each row, find the transition point
    # A row is sorted if there's no 0 after a 1
    # result[:, k] >= result[:, k+1] would be bad... wait, sorted ascending means result[:, k] <= result[:, k+1]
    # For Boolean: 0 <= 1 is fine, 1 <= 0 is not
    # So check: no position where result[:, k] = 1 and result[:, k+1] = 0
    
    # Vectorized check: for k in 0..N-2, check if result[:, k] == 1 and result[:, k+1] == 0
    for k in range(N - 1):
        bad = (result[:, k] == 1) & (result[:, k+1] == 0)
        if np.any(bad):
            return False
    return True

def _get_baseline_network():
    """Odd-even transposition sorting network: 120 comparators for 16 wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _try_remove_comparators(network, max_time):
    """Try to remove comparators one at a time, keeping the network valid.
    Returns the smallest valid network found."""
    best = list(network)
    best_size = len(best)
    
    # Try removing each comparator
    for idx in range(len(best) - 1, -1, -1):
        if time.time() > max_time:
            break
        candidate = best[:idx] + best[idx+1:]
        if _is_sorting_network(candidate):
            best = candidate
            best_size = len(best)
    
    return best

def _try_add_and_remove(network, max_time):
    """Try replacing comparators to reduce count.
    Strategy: try adding a new comparator and then removing more than one old one."""
    best = list(network)
    best_size = len(best)
    
    # Try all possible comparator positions
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # For each possible new comparator, try adding it and then removing old ones
    for new_comp in all_pairs:
        if time.time() > max_time:
            break
        # Add the new comparator at the end
        candidate = list(best) + [new_comp]
        
        # Try removing each old comparator
        for idx in range(len(candidate) - 1, -1, -1):
            if time.time() > max_time:
                break
            if candidate[idx] == new_comp:
                continue  # Don't remove the one we just added
            reduced = candidate[:idx] + candidate[idx+1:]
            if len(reduced) < best_size and _is_sorting_network(reduced):
                best = reduced
                best_size = len(best)
    
    return best

def _local_search(network, max_time):
    """Local search: try swapping, removing, and adding comparators."""
    best = list(network)
    best_size = len(best)
    
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    improved = True
    while improved and time.time() < max_time:
        improved = False
        
        # Try removing
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > max_time:
                break
            candidate = best[:idx] + best[idx+1:]
            if _is_sorting_network(candidate):
                best = candidate
                best_size = len(best)
                improved = True
                break
        
        if time.time() > max_time:
            break
        
        # Try replacing: remove one, add one
        for remove_idx in range(len(best)):
            if time.time() > max_time:
                break
            for add_comp in all_pairs:
                if time.time() > max_time:
                    break
                if add_comp in best:
                    continue
                candidate = best[:remove_idx] + best[remove_idx+1:] + [add_comp]
                if len(candidate) < best_size and _is_sorting_network(candidate):
                    best = candidate
                    best_size = len(best)
                    improved = True
                    break
            if improved:
                break
    
    return best

def construct():
    """Construct a sorting network with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 78  # 78s to leave safety margin
    
    # Start with baseline
    network = _get_baseline_network()
    best = list(network)
    best_size = len(best)
    
    # Phase 1: Greedy removal
    best = _try_remove_comparators(best, deadline)
    best_size = len(best)
    
    if time.time() > deadline:
        return best
    
    # Phase 2: Local search with replacements
    best = _local_search(best, deadline)
    best_size = len(best)
    
    return best
