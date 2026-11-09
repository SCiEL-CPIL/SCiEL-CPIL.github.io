"""
Size-efficient sorting network on 18 wires.
Strategy: Start from a known-good baseline (odd-even transposition), then use
local search (add, remove, swap, relocate comparators) with a fast
bit-parallel verifier over all 2^18 Boolean inputs to minimize comparator count.
A deadline guard ensures termination before the 90s wall-clock limit.
"""

import time
import random
import numpy as np

N = 18
NUM_INPUTS = 1 << N  # 262144

# Precompute masks for fast application of comparators
# For a comparator (i, j), we need to set bit i to min(x_i, x_j) and bit j to max(x_i, x_j).
# In the context of 0/1 values (bits), min(a,b) = a & b and max(a,b) = a | b.
# So for a state x (an integer representing the vector):
# bit_i_new = bit_i & bit_j
# bit_j_new = bit_i | bit_j
# We can precompute masks to extract bits and then recombine.

# However, applying a comparator to a numpy array of 262144 integers is fast.
# Let's represent the state as a numpy array of uint32 or uint64.
# Actually, since we are checking all 2^18 inputs, we can represent the "state" for all inputs simultaneously.
# Each input is a vector of 18 bits.
# We can represent the current value of each wire for all inputs as a numpy array of shape (18, 262144) of uint8?
# Or better, represent the 262144 inputs as rows in a matrix of bits.
# Let X be a matrix of shape (262144, 18) where X[k, i] is the value of wire i for input k.
# Initially, X is the identity matrix of bits? No, X[k, i] is bit i of k.
# We can generate X using numpy bit operations.

# Let's try a different representation for speed:
# A state is a tuple of 18 numpy arrays, each of shape (262144,) containing 0s and 1s.
# Or a single array of shape (18, 262144).
# Let's use a single array `state` of shape (18, 262144) with dtype uint8.
# state[i, k] is the value of wire i for input k.

def _generate_initial_state():
    """Generate the initial state matrix for all 2^18 inputs."""
    # We need an array of indices 0 to 2^18 - 1.
    indices = np.arange(1 << N, dtype=np.uint32)
    # We want state[i, k] = (indices[k] >> i) & 1
    # We can do this efficiently.
    state = np.zeros((N, 1 << N), dtype=np.uint8)
    for i in range(N):
        state[i] = (indices >> i) & 1
    return state

def _apply_comparator(state, i, j):
    """
    Apply comparator (i, j) to the state matrix.
    min goes to i, max goes to j.
    For 0/1 values: min = AND, max = OR.
    """
    # We need to update state[i] and state[j].
    # new_i = state[i] & state[j]
    # new_j = state[i] | state[j]
    # Note: we must use the original values for both calculations.
    orig_i = state[i].copy()
    orig_j = state[j].copy()
    state[i] = orig_i & orig_j
    state[j] = orig_i | orig_j

def _is_sorted(state):
    """
    Check if the state is sorted for all inputs.
    Sorted means for all k, state[0, k] <= state[1, k] <= ... <= state[N-1, k].
    Since values are 0/1, this means once a 1 appears, all subsequent wires must be 1.
    Equivalently, for each k, the vector state[:, k] is non-decreasing.
    This is equivalent to: for all i, state[i, k] <= state[i+1, k].
    """
    # Check if any state[i] > state[i+1] for any i, k.
    # state[i] > state[i+1] is equivalent to state[i] & ~state[i+1] != 0.
    # Or simply np.any(state[:-1] > state[1:])
    return not np.any(state[:-1] > state[1:])

def _evaluate_network(network):
    """
    Evaluate a network (list of (i, j) pairs).
    Returns (is_valid, comparator_count).
    """
    if not network:
        return False, 0
    state = _generate_initial_state()
    for i, j in network:
        _apply_comparator(state, i, j)
    valid = _is_sorted(state)
    return valid, len(network)

# --- Local Search ---

def _random_network():
    """Generate a random valid-ish network (might not sort, but good starting point for search)."""
    # Start with a few comparators
    net = []
    for _ in range(20):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        net.append((i, j))
    return net

def _mutate_network(net):
    """
    Perform a random mutation on the network.
    Returns a new network.
    """
    net = list(net)
    op = random.choice(['add', 'remove', 'swap', 'relocate'])
    
    if op == 'add' and len(net) < 4096:
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(net))
        net.insert(pos, (i, j))
        
    elif op == 'remove' and len(net) > 1:
        pos = random.randint(0, len(net) - 1)
        net.pop(pos)
        
    elif op == 'swap' and len(net) >= 2:
        i1 = random.randint(0, len(net) - 1)
        i2 = random.randint(0, len(net) - 1)
        net[i1], net[i2] = net[i2], net[i1]
        
    elif op == 'relocate' and len(net) >= 1:
        i1 = random.randint(0, len(net) - 1)
        i2 = random.randint(0, len(net) - 1)
        net[i1], net[i2] = net[i2], net[i1]
        # Or just move one element
        # Let's do a proper relocate: pick an element, remove it, insert at new pos
        elem = net.pop(i1)
        pos = random.randint(0, len(net))
        net.insert(pos, elem)
        
    return net

def construct():
    """
    Construct a sorting network.
    Uses local search to minimize comparator count.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Baseline: Odd-even transposition
    def _baseline_network():
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Initialize with baseline
    current_net = _baseline_network()
    current_valid, current_count = _evaluate_network(current_net)
    
    best_net = list(current_net)
    best_count = current_count
    best_valid = current_valid
    
    # If baseline is invalid (shouldn't be, but just in case), try random
    if not best_valid:
        for _ in range(100):
            net = _random_network()
            valid, count = _evaluate_network(net)
            if valid:
                best_net = net
                best_count = count
                best_valid = True
                break
        if not best_valid:
            # Return a known valid network if we can't find one?
            # The baseline is always valid.
            best_net = _baseline_network()
            best_valid, best_count = _evaluate_network(best_net)

    # Local search
    # We want to minimize count.
    # We only accept moves that result in a valid network.
    # If we remove a comparator and it becomes invalid, we reject.
    # If we add a comparator, it might become valid (if it wasn't) or stay valid.
    # We prefer fewer comparators.
    
    # To speed up, we can cache the state after each step?
    # No, state is huge (18 * 262144). Copying is expensive.
    # But applying comparators is O(N * 2^N).
    # 18 * 262144 = ~4.7M operations per comparator application.
    # In NumPy, this is very fast, maybe ~1ms.
    # So evaluating a network of size M takes M * 1ms.
    # If M is ~100, that's 100ms.
    # We have 85 seconds. So we can do ~850 evaluations.
    # This is limited. We need to be smarter or accept fewer iterations.
    # Or optimize the evaluation.
    
    # Optimization:
    # Instead of regenerating state from scratch, we can keep the state.
    # But if we revert, we need the previous state.
    # Storing states (18 * 262144 bytes = 4.7 MB) is feasible.
    # We can store the state history for backtracking.
    
    # Let's implement a simple hill climbing with state caching.
    
    state = _generate_initial_state()
    
    # Apply current_net to state
    for i, j in current_net:
        _apply_comparator(state, i, j)
    
    current_valid = _is_sorted(state)
    current_count = len(current_net)
    
    # We track the best valid network found so far
    best_net = list(current_net)
    best_count = current_count
    best_state = state.copy()
    
    # Local search loop
    while time.time() < deadline:
        # Generate a neighbor
        neighbor_net = _mutate_network(current_net)
        
        # Check if neighbor is valid
        # We need to re-evaluate from scratch or from a cached state?
        # Since the mutation can be anywhere, re-evaluating from scratch is safest.
        # But it's slow.
        # Alternative: Only try removing comparators from the end?
        # Or try swapping adjacent comparators?
        
        # Let's try a simpler strategy:
        # 1. Try to remove a random comparator. If still valid, accept.
        # 2. Try to swap two adjacent comparators. If still valid, accept (count same, might enable further removals).
        # 3. Try to add a comparator? Only if we are stuck and count is high? No, adding increases count.
        
        # Strategy: Focus on removal and reordering.
        
        # Try removal
        if len(current_net) > 1:
            idx = random.randint(0, len(current_net) - 1)
            test_net = current_net[:idx] + current_net[idx+1:]
            # Evaluate test_net
            # To save time, we can stop early if it becomes invalid?
            # No, we need to know if it's valid at the end.
            
            # Optimization: If we remove a comparator, we can check if the network is still valid.
            # We can do this by running the network.
            # Let's just run it.
            
            valid, count = _evaluate_network(test_net)
            if valid:
                # Accept
                current_net = test_net
                current_count = count
                if current_count < best_count:
                    best_count = current_count
                    best_net = list(current_net)
                    # Update best_state? We don't strictly need it unless we backtrack.
                    # But we need the state for the next mutation if we keep it.
                    # Let's re-apply to get the state.
                    state = _generate_initial_state()
                    for i, j in current_net:
                        _apply_comparator(state, i, j)
                continue
            
        # Try swapping adjacent comparators
        if len(current_net) >= 2:
            idx = random.randint(0, len(current_net) - 2)
            test_net = list(current_net)
            test_net[idx], test_net[idx+1] = test_net[idx+1], test_net[idx]
            
            valid, count = _evaluate_network(test_net)
            if valid:
                # Count is same, but structure changed. Might allow future removals.
                # Accept it? It doesn't improve count immediately.
                # Let's accept it with some probability to explore.
                if random.random() < 0.5:
                    current_net = test_net
                    # Update state
                    state = _generate_initial_state()
                    for i, j in current_net:
                        _apply_comparator(state, i, j)
                continue
        
        # If stuck, try a random restart?
        # Or try adding a comparator to enable a different structure?
        # Unlikely to help reduce count.
        
        # Let's try a more aggressive move: relocate a comparator.
        if len(current_net) >= 2:
            idx1 = random.randint(0, len(current_net) - 1)
            idx2 = random.randint(0, len(current_net) - 1)
            if idx1 != idx2:
                test_net = list(current_net)
                elem = test_net.pop(idx1)
                # Adjust idx2 if necessary
                if idx1 < idx2:
                    idx2 -= 1
                test_net.insert(idx2, elem)
                
                valid, count = _evaluate_network(test_net)
                if valid:
                    if random.random() < 0.5:
                        current_net = test_net
                        state = _generate_initial_state()
                        for i, j in current_net:
                            _apply_comparator(state, i, j)
                    continue

    return best_net

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We need to ensure the interface is exactly as requested.
# The prompt says: "Keep the function names and signatures of the seed program."
# Seed program had: `def construct():`
# So we just need to make sure `construct` is the entry point.
# The code above defines `construct`.
# We should remove the EVOLVE-BLOCK markers if they are not required, or keep them if the system expects them.
# The prompt says: "Output the COMPLETE program in a single 