"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 18
_SEED_COMPARATORS = 153

def _apply_comparator(state, i, j):
    """Apply a compare-exchange operation to a state array."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _vectorized_sort_check(comparators, inputs):
    """
    Vectorized check: apply all comparators to a batch of input vectors.
    Returns a boolean array of shape (batch,) indicating whether each
    input was sorted correctly.
    """
    batch = inputs.shape[0]
    states = inputs.copy()  # (batch, N)
    
    for i, j in comparators:
        # Compare wire i and wire j across all batch entries
        # min goes to i, max goes to j
        min_val = np.minimum(states[:, i], states[:, j])
        max_val = np.maximum(states[:, i], states[:, j])
        states[:, i] = min_val
        states[:, j] = max_val
    
    # Check if each row is sorted
    # A row is sorted if state[:, k] <= state[:, k+1] for all k
    diffs = np.diff(states, axis=1)
    sorted_mask = np.all(diffs >= 0, axis=1)
    return sorted_mask

def _is_sorted_network(comparators, test_inputs=None):
    """Verify that a network of comparators sorts all Boolean inputs."""
    if test_inputs is None:
        # Generate all 2^N Boolean inputs as a (2^N, N) array
        bits = np.arange(2**N, dtype=np.int32)
        test_inputs = ((bits[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int32)
    
    result = _vectorized_sort_check(comparators, test_inputs)
    return bool(np.all(result))

def _count_comparators(comparators):
    """Count the number of comparators in a network."""
    return len(comparators)

def _depth(comparators):
    """Calculate the parallel depth of a sorting network."""
    if not comparators:
        return 0
    
    wire_depth = np.zeros(N, dtype=int)
    depth = 0
    for i, j in comparators:
        wire_depth[i] = max(wire_depth[i], depth)
        wire_depth[j] = max(wire_depth[j], depth)
        depth = max(wire_depth[i], wire_depth[j]) + 1
    
    return int(depth)

def _build_seed_network():
    """Build the odd-even transposition sorting network."""
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _local_search_remove(comparators, all_inputs, time_limit):
    """
    Greedily remove redundant comparators one at a time.
    Returns the reduced network.
    """
    network = list(comparators)
    changed = True
    while changed and time.time() < time_limit:
        changed = False
        for idx in range(len(network) - 1, -1, -1):
            candidate = network[:idx] + network[idx + 1:]
            if _is_sorted_network(candidate, all_inputs):
                network = candidate
                changed = True
                break
    return network

def construct():
    """Construct a minimal sorting network using vectorized verification + local search."""
    start_time = time.time()
    deadline = 80.0  # Safety margin before 90s limit
    
    # Pre-generate all 2^18 = 262144 Boolean inputs
    bits = np.arange(2**N, dtype=np.int32)
    all_inputs = ((bits[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int32)
    
    # Start with the seed odd-even transposition network
    best_network = _build_seed_network()
    best_count = len(best_network)
    
    # Phase 1: Greedy removal of redundant comparators
    best_network = _local_search_remove(best_network, all_inputs, min(deadline, start_time + 30))
    best_count = len(best_network)
    
    # Phase 2: Simulated annealing / local search with swaps and insertions
    # Try to find a better network by perturbing and re-optimizing
    current_network = list(best_network)
    current_count = best_count
    
    # Use a subset of inputs for fast intermediate checks
    # Full check only when we think we have an improvement
    sample_size = min(4096, 2**N)
    np.random.seed(42)
    sample_indices = np.random.choice(2**N, size=sample_size, replace=False)
    sample_inputs = all_inputs[sample_indices]
    
    def _fast_check(comparators):
        """Quick check on sample inputs."""
        return _is_sorted_network(comparators, sample_inputs)
    
    def _full_check(comparators):
        """Full verification on all inputs."""
        return _is_sorted_network(comparators, all_inputs)
    
    # Local search moves: remove, swap, insert
    rng = np.random.RandomState(123)
    temperature = 5.0
    cooling = 0.95
    min_temp = 0.1
    
    while time.time() < deadline and temperature > min_temp:
        move_count = 0
        max_moves = 200  # moves per temperature level
        
        for _ in range(max_moves):
            if time.time() > deadline:
                break
            if temperature < min_temp:
                break
            
            # Choose a random move
            move_type = rng.randint(0, 3)
            candidate = list(current_network)
            
            if move_type == 0 and len(candidate) > 10:
                # Remove a random comparator
                idx = rng.randint(0, len(candidate))
                candidate.pop(idx)
            elif move_type == 1 and len(candidate) > 1:
                # Swap two adjacent comparators
                idx = rng.randint(0, len(candidate) - 1)
                candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
            elif move_type == 2:
                # Insert a random comparator at a random position
                i = rng.randint(0, N - 1)
                j = rng.randint(i + 1, N)
                pos = rng.randint(0, len(candidate) + 1)
                candidate.insert(pos, (i, j))
            else:
                continue
            
            move_count += 1
            
            # Fast check first
            if not _fast_check(candidate):
                continue
            
            # Full check
            if _full_check(candidate):
                cand_count = len(candidate)
                curr_count = len(current_network)
                
                # Accept if better or with probability based on temperature
                if cand_count < curr_count:
                    current_network = candidate
                    current_count = cand_count
                    if cand_count < best_count:
                        best_network = candidate
                        best_count = cand_count
                else:
                    # Metropolis acceptance for equal or worse (allows exploration)
                    delta = cand_count - curr_count
                    if delta <= 0 or rng.random() < np.exp(-delta / temperature):
                        current_network = candidate
                        current_count = cand_count
        
        temperature *= cooling
    
    # Final greedy cleanup
    if time.time() < deadline - 2:
        best_network = _local_search_remove(best_network, all_inputs, deadline - 1)
    
    return best_network
# EVOLVE-BLOCK-END
