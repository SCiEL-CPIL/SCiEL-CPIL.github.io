"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17

# EVOLVE-BLOCK-START
def _simulate_network_fast(comparators, input_vec):
    """Simulate a sorting network on a single input vector.
    Returns True if output is sorted (non-decreasing)."""
    x = input_vec.copy()
    for i, j in comparators:
        if x[i] > x[j]:
            x[i], x[j] = x[j], x[i]
    # Check sorted
    for k in range(N - 1):
        if x[k] > x[k + 1]:
            return False
    return True

def _is_sorted_network(comparators):
    """Check if a network sorts all 2^N Boolean inputs using the 0-1 principle."""
    n_bools = 1 << N
    # Use vectorized check: represent all 2^N inputs as a (n_bools, N) array
    # But that's 131072 * 17 = ~2.2M elements, manageable.
    # Actually, let's be smarter: process all inputs in parallel using bit operations.
    
    # Represent each of the 2^N inputs as a row in an array of shape (2^N, N)
    # where entry [k, w] is the value on wire w for input k.
    # Input k has bit w = (k >> w) & 1
    
    # We'll process comparators one at a time on the whole array.
    # For each comparator (i, j): for each row, if arr[row, i] > arr[row, j], swap.
    
    # Create the initial array: shape (2^N, N)
    # arr[k, w] = (k >> w) & 1
    indices = np.arange(n_bools, dtype=np.uint8)
    # For wire w, value is (indices >> w) & 1
    # Build column by column
    arr = np.zeros((n_bools, N), dtype=np.uint8)
    for w in range(N):
        arr[:, w] = (indices >> w) & 1
    
    for i, j in comparators:
        # Where arr[:, i] > arr[:, j], swap
        mask = arr[:, i] > arr[:, j]
        if np.any(mask):
            arr[mask, i] = 1
            arr[mask, j] = 0
    
    # Check if all rows are sorted: arr[:, k] <= arr[:, k+1] for all k
    for k in range(N - 1):
        if np.any(arr[:, k] > arr[:, k + 1]):
            return False
    return True

def _depth_of_network(comparators):
    """Compute the parallel depth of a network."""
    # Depth = longest chain of dependent comparators
    # A comparator (i,j) at position p has depth = 1 + max(depth of prior comparators touching i or j)
    wire_depth = [0] * N
    max_depth = 0
    for i, j in comparators:
        d = 1 + max(wire_depth[i], wire_depth[j])
        wire_depth[i] = d
        wire_depth[j] = d
        max_depth = max(max_depth, d)
    return max_depth

def construct():
    """Construct a sorting network on 17 wires with as few comparators as possible."""
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill
    
    random.seed(42)
    
    # Start with the odd-even transposition network as seed
    seed_network = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    best_network = seed_network
    best_count = len(seed_network)
    
    # Verify seed is correct
    if not _is_sorted_network(seed_network):
        # This shouldn't happen, but just in case
        return seed_network
    
    # Local search: randomly remove comparators and check if still sorting
    # We iterate: try to remove one comparator at a time, keep if still valid
    # This is a greedy reduction approach
    
    current = list(seed_network)
    current_count = len(current)
    
    # Multiple passes of greedy removal
    improved = True
    pass_num = 0
    while improved and time.time() < deadline:
        improved = False
        pass_num += 1
        # Shuffle order of removal attempts for variety
        indices = list(range(len(current)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 2.0:
                break
            if idx >= len(current):
                continue
            
            # Try removing comparator at idx
            candidate = current[:idx] + current[idx+1:]
            if not candidate:
                break
            
            # Quick check: is it still a valid sorting network?
            if _is_sorted_network(candidate):
                current = candidate
                current_count -= 1
                improved = True
                if current_count < best_count:
                    best_count = current_count
                    best_network = list(current)
                # After removal, re-shuffle remaining indices
                indices = list(range(len(current)))
                random.shuffle(indices)
        
        if time.time() > deadline - 2.0:
            break
    
    # If we have time, try a more aggressive search: 
    # Randomly perturb the network and see if we can find a better one
    if time.time() < deadline - 5.0:
        # Try some random restarts with perturbation
        for restart in range(50):
            if time.time() > deadline - 3.0:
                break
            
            # Start from current best and try random modifications
            candidate = list(best_network)
            
            # Try removing random subsets
            for _ in range(20):
                if time.time() > deadline - 2.0:
                    break
                if not candidate:
                    break
                idx = random.randint(0, len(candidate) - 1)
                trial = candidate[:idx] + candidate[idx+1:]
                if trial and _is_sorted_network(trial):
                    candidate = trial
                    if len(candidate) < best_count:
                        best_count = len(candidate)
                        best_network = list(candidate)
    
    # Ensure we return a valid network
    if not best_network or not _is_sorted_network(best_network):
        return seed_network
    
    return best_network
# EVOLVE-BLOCK-END
