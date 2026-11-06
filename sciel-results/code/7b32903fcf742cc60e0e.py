import time
import numpy as np

N = 16

# Precompute all 2^16 zero-one inputs as a numpy array of shape (65536, 16)
# Each row is a distinct binary vector of length 16.
# Using vectorized bit extraction.
bits = np.arange(2**N, dtype=np.uint16)[:, None]
masks = (1 << np.arange(N, dtype=np.uint16))[None, :]
all_inputs = (bits & masks).astype(np.uint8)

# Precompute sorted check mask: for each input, check if it is sorted (non-decreasing)
# sorted_mask: shape (65536,) boolean, True if row is sorted
# A row is sorted if all adjacent pairs satisfy state[i] <= state[i+1]
# We can check this efficiently.
# Actually, we will compute sortedness on the fly after simulation for flexibility,
# but precomputing the "already sorted" inputs might help? No, we need to check output.
# Let's just precompute the comparison indices for speed.
# We need to check: out[:, 0] <= out[:, 1] <= ... <= out[:, 15]
# This is equivalent to: np.all(out[:, :-1] <= out[:, 1:], axis=1)

def simulate_network(network, inputs):
    """
    Simulate the sorting network on the given inputs.
    network: list of (i, j) tuples
    inputs: numpy array of shape (65536, 16)
    Returns: the output array after all comparators.
    """
    if not network:
        return inputs.copy()
    
    state = inputs.copy()
    for i, j in network:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    return state

def is_valid(network):
    """
    Check if the network sorts all zero-one inputs.
    """
    if not network:
        # Empty network only works if all inputs are already sorted, which is false.
        # But we can just simulate.
        pass
        
    out = simulate_network(network, all_inputs)
    return np.all(out[:, :-1] <= out[:, 1:], axis=1)

def is_valid_fast(network):
    """
    Faster validity check with early exit if possible?
    Numpy operations are vectorized, so early exit per-row is hard.
    But we can check a subset first?
    Let's stick to full check but ensure simulation is fast.
    """
    out = simulate_network(network, all_inputs)
    return np.all(out[:, :-1] <= out[:, 1:], axis=1)

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s safety margin before 90s kill

    # Start with a known valid network: Odd-Even Transposition
    # 16 wires, 16 phases.
    # Phase k: comparators (i, i+1) for i in {k%2, k%2+2, ...}
    initial_network = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            initial_network.append((i, i + 1))
    
    best_network = initial_network
    best_count = len(best_network)
    
    # 1. Greedy Removal
    # Try to remove each comparator one by one from the end.
    # If removal preserves validity, keep it removed.
    # Iterate until no more removals are possible.
    changed = True
    while changed and time.time() < deadline:
        changed = False
        # Iterate backwards to handle index shifts easily
        for idx in range(len(best_network) - 1, -1, -1):
            if time.time() > deadline:
                break
            candidate = best_network[:idx] + best_network[idx+1:]
            if is_valid_fast(candidate):
                best_network = candidate
                best_count = len(best_network)
                changed = True
                # Restart loop since indices changed
                break
        
    # 2. Local Search: Swap and Move operations
    # We can swap the order of two comparators.
    # Or move a comparator to a different position.
    # Swapping is safer as it maintains the set of comparators.
    # Moving might allow further removals.
    
    attempts = 0
    max_attempts = 50000
    last_improvement_time = time.time()
    
    while time.time() < deadline and attempts < max_attempts:
        attempts += 1
        if len(best_network) < 2:
            break
            
        # Randomly pick two indices
        i = np.random.randint(0, len(best_network))
        j = np.random.randint(0, len(best_network))
        if i == j:
            continue
        
        candidate = list(best_network)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        
        if is_valid_fast(candidate):
            best_network = candidate
            last_improvement_time = time.time()
            
            # Try greedy removals after a successful swap
            changed = True
            while changed and time.time() < deadline:
                changed = False
                for idx in range(len(best_network) - 1, -1, -1):
                    if time.time() > deadline:
                        break
                    cand = best_network[:idx] + best_network[idx+1:]
                    if is_valid_fast(cand):
                        best_network = cand
                        best_count = len(best_network)
                        changed = True
                        break
        else:
            # Simulated Annealing style: accept worse with small prob?
            # No, we must maintain validity. Only accept valid swaps.
            # To escape local minima, we might need to temporarily add a comparator?
            # Or use a different move: Move a comparator to a random new position.
            if np.random.random() < 0.1:
                # Move operation
                pos = np.random.randint(0, len(best_network))
                comp = best_network[pos]
                new_pos = np.random.randint(0, len(best_network))
                candidate = list(best_network)
                candidate.pop(pos)
                candidate.insert(new_pos, comp)
                if is_valid_fast(candidate):
                    best_network = candidate
                    last_improvement_time = time.time()
                    # Greedy removals
                    changed = True
                    while changed and time.time() < deadline:
                        changed = False
                        for idx in range(len(best_network) - 1, -1, -1):
                            if time.time() > deadline:
                                break
                            cand = best_network[:idx] + best_network[idx+1:]
                            if is_valid_fast(cand):
                                best_network = cand
                                best_count = len(best_network)
                                changed = True
                                break

    return best_network
