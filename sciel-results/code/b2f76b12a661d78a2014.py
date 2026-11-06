"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 18
TIME_LIMIT = 80.0  # seconds, safety margin before 90s kill

# EVOLVE-BLOCK-START

def _make_wires(state):
    """Extract wire values from boolean state (list of ints 0/1)."""
    return [state[i] for i in range(N)]

def _apply_compare(wires, i, j):
    """Apply compare-exchange: min goes to wire i, max to wire j."""
    if wires[i] > wires[j]:
        wires[i], wires[j] = wires[j], wires[i]

def _is_sorted(wires):
    """Check if wires are in non-decreasing order."""
    for i in range(N - 1):
        if wires[i] > wires[i + 1]:
            return False
    return True

def _verify_network(network, states):
    """Verify network sorts all given states. Returns True if all sorted."""
    for state in states:
        wires = _make_wires(state)
        for (i, j) in network:
            _apply_compare(wires, i, j)
        if not _is_sorted(wires):
            return False
    return True

def _verify_all_boolean(network):
    """Verify over all 2^18 Boolean inputs. Returns True if valid."""
    # Use numpy for efficient checking
    M = 1 << N
    # states[k] = bit representation of k
    # We'll process in batches using the network
    # For speed, use the 0-1 principle: only need to check that
    # the sorted output of each Boolean input is non-decreasing.
    
    # Fast vectorized approach:
    # For each of 2^18 inputs, simulate the network.
    # Use bit-level operations on numpy arrays.
    
    # Input states: M x N matrix where states[k, i] = bit i of k
    k_arr = np.arange(M, dtype=np.uint64)
    # Build states matrix: states[k, i] = (k >> i) & 1
    states = np.zeros((M, N), dtype=np.uint8)
    for i in range(N):
        states[:, i] = ((k_arr >> i) & 1).astype(np.uint8)
    
    # Simulate network
    wires = states.copy()  # M x N
    for (i, j) in network:
        # min goes to i, max goes to j
        wi = wires[:, i]
        wj = wires[:, j]
        # min = wi & wj, max = wi | wj (for bits)
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
    
    # Check sorted: wires[:, i] <= wires[:, i+1] for all i
    for i in range(N - 1):
        if np.any(wires[:, i] > wires[:, i + 1]):
            return False
    return True

def _verify_subset(network, num_samples=4096):
    """Quick verification on a random subset of Boolean inputs."""
    M = 1 << N
    k_arr = np.random.randint(0, M, size=num_samples).astype(np.uint64)
    states = np.zeros((num_samples, N), dtype=np.uint8)
    for i in range(N):
        states[:, i] = ((k_arr >> i) & 1).astype(np.uint8)
    
    wires = states.copy()
    for (i, j) in network:
        wi = wires[:, i]
        wj = wires[:, j]
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
    
    for i in range(N - 1):
        if np.any(wires[:, i] > wires[:, i + 1]):
            return False
    return True

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network for N wires."""
    network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            network.append((i, i + 1))
    return network

def _count_sort_network():
    """Counting sort network: for each pair, compare."""
    network = []
    for i in range(N):
        for j in range(i + 1, N):
            network.append((i, j))
    return network

def _random_network(num_comparators):
    """Generate a random sorting network (likely invalid but for exploration)."""
    network = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 1)
        j = random.randint(i + 1, N - 1)
        network.append((i, j))
    return network

def _remove_redundant(network):
    """Try to remove redundant comparators from a valid network."""
    network = list(network)
    removed = []
    changed = True
    while changed:
        changed = False
        # Try removing each comparator
        for idx in range(len(network)):
            if idx in removed:
                continue
            candidate = network[:idx] + network[idx+1:]
            if _verify_subset(candidate, num_samples=2048):
                # More thorough check
                if _verify_all_boolean(candidate):
                    network = candidate
                    removed = []
                    changed = True
                    break
        # If we made a change, reset removed indices
        if changed:
            removed = []
    return network

def _greedy_minimize(network, time_budget=30.0):
    """Greedy removal of redundant comparators with time budget."""
    start_time = time.time()
    network = list(network)
    
    # First pass: try removing each comparator
    while time.time() - start_time < time_budget:
        improved = False
        for idx in range(len(network)):
            if time.time() - start_time > time_budget:
                break
            candidate = network[:idx] + network[idx+1:]
            # Quick check on subset
            if _verify_subset(candidate, num_samples=1024):
                # Full verification
                if _verify_all_boolean(candidate):
                    network = candidate
                    improved = True
                    break
        if not improved:
            break
    
    return network

def _swap_local_search(network, time_budget=15.0):
    """Local search by swapping adjacent comparators."""
    start_time = time.time()
    network = list(network)
    best_len = len(network)
    
    while time.time() - start_time < time_budget:
        improved = False
        # Try swapping adjacent comparators
        for idx in range(len(network) - 1):
            if time.time() - start_time > time_budget:
                break
            candidate = network.copy()
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            if _verify_subset(candidate, num_samples=1024):
                if _verify_all_boolean(candidate):
                    network = candidate
                    improved = True
                    break
        if not improved:
            break
    
    return network

def _error_guided_repair(network, time_budget=15.0):
    """Try to repair a network that fails on some inputs by adding comparators."""
    start_time = time.time()
    network = list(network)
    
    # Find failing inputs
    M = 1 << N
    k_arr = np.arange(M, dtype=np.uint64)
    states = np.zeros((M, N), dtype=np.uint8)
    for i in range(N):
        states[:, i] = ((k_arr >> i) & 1).astype(np.uint8)
    
    wires = states.copy()
    for (i, j) in network:
        wi = wires[:, i]
        wj = wires[:, j]
        wires[:, i] = np.minimum(wi, wj)
        wires[:, j] = np.maximum(wi, wj)
    
    # Check which inputs fail
    fails = np.zeros(M, dtype=bool)
    for i in range(N - 1):
        fails |= (wires[:, i] > wires[:, i + 1])
    
    if not np.any(fails):
        return network
    
    # Try adding comparators to fix failing inputs
    failing_indices = np.where(fails)[0]
    
    while time.time() - start_time < time_budget and np.any(fails):
        improved = False
        # Try adding each possible comparator
        for i in range(N):
            for j in range(i + 1, N):
                if (i, j) in network:
                    continue
                candidate = network + [(i, j)]
                if _verify_subset(candidate, num_samples=2048):
                    if _verify_all_boolean(candidate):
                        network = candidate
                        improved = True
                        # Re-check fails
                        wires2 = states.copy()
                        for (a, b) in network:
                            wi2 = wires2[:, a]
                            wj2 = wires2[:, b]
                            wires2[:, a] = np.minimum(wi2, wj2)
                            wires2[:, b] = np.maximum(wi2, wj2)
                        fails = np.zeros(M, dtype=bool)
                        for a in range(N - 1):
                            fails |= (wires2[:, a] > wires2[:, a + 1])
                        break
            if improved:
                break
        if not improved:
            break
    
    return network

def construct():
    start_time = time.time()
    
    # Start with odd-even transposition network
    network = _odd_even_transposition()
    print(f"Initial network: {len(network)} comparators")
    
    # Verify it's valid
    if not _verify_all_boolean(network):
        # Fallback: use counting sort
        network = _count_sort_network()
    
    # Phase 1: Greedy minimization
    time_left = TIME_LIMIT - (time.time() - start_time)
    if time_left > 10:
        network = _greedy_minimize(network, time_budget=min(time_left - 5, 30.0))
        print(f"After greedy minimization: {len(network)} comparators")
    
    # Phase 2: Swap local search
    time_left = TIME_LIMIT - (time.time() - start_time)
    if time_left > 10:
        network = _swap_local_search(network, time_budget=min(time_left - 5, 15.0))
        print(f"After swap local search: {len(network)} comparators")
    
    # Phase 3: Another round of greedy minimization
    time_left = TIME_LIMIT - (time.time() - start_time)
    if time_left > 10:
        network = _greedy_minimize(network, time_budget=min(time_left - 5, 20.0))
        print(f"After second greedy minimization: {len(network)} comparators")
    
    # Final verification
    if not _verify_all_boolean(network):
        # Fallback to known valid network
        network = _odd_even_transposition()
    
    # Ensure we're within time limit
    elapsed = time.time() - start_time
    if elapsed > TIME_LIMIT - 5:
        print(f"Time critical: {elapsed:.1f}s elapsed")
    
    print(f"Final network: {len(network)} comparators, {elapsed:.1f}s elapsed")
    return network

# EVOLVE-BLOCK-END
