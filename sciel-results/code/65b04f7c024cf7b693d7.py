"""Size-efficient sorting network for 18 wires.
Uses a hybrid approach:
1. Start with a known good network (odd-even transposition, 153 comparators).
2. Apply aggressive local search: comparator deletion (with bit-parallel verification)
   and comparator swapping/mutation to reduce count.
3. Use a time budget to explore as many improvements as possible.
"""
import time
import random
import numpy as np

N = 18
NUM_BOOL_INPUTS = 1 << N  # 262144

# Precompute all boolean inputs as a uint8 array of shape (NUM_BOOL_INPUTS, N)
# This is 262144 * 18 = 4.7 MB, manageable.

def _precompute_inputs():
    """Return a numpy array of shape (262144, 18) with all boolean inputs."""
    # Generate all 2^18 bit patterns
    arr = np.zeros((1 << N, N), dtype=np.uint8)
    for i in range(N):
        # Bit i of each row
        arr[:, i] = (np.arange(1 << N) >> i) & 1
    return arr

def _apply_comparator(inputs, i, j):
    """Apply compare-exchange (i, j) to all inputs. Returns new array.
    min goes to wire i, max goes to wire j."""
    a = inputs[:, i].copy()
    b = inputs[:, j].copy()
    # min = min(a, b), max = max(a, b)
    new_i = np.minimum(a, b)
    new_j = np.maximum(a, b)
    # We need to update the input array
    inputs[:, i] = new_i
    inputs[:, j] = new_j
    return inputs

def _is_sorted(inputs):
    """Check if all rows are sorted (non-decreasing)."""
    # A row is sorted if for all k, inputs[k, m] <= inputs[k, m+1]
    # i.e., inputs[:, :-1] <= inputs[:, 1:]
    return np.all(inputs[:, :-1] <= inputs[:, 1:])

def _verify_network(comparators):
    """Verify that the network sorts all boolean inputs.
    Returns True if sorted, False otherwise."""
    if not comparators:
        return True
    # Use a copy of the inputs
    inputs = _INPUTS.copy()
    for (i, j) in comparators:
        _apply_comparator(inputs, i, j)
    return _is_sorted(inputs)

def _verify_network_fast(comparators, inputs):
    """Apply comparators to a copy of inputs and check sortedness.
    Returns True if sorted."""
    inp = inputs.copy()
    for (i, j) in comparators:
        a = inp[:, i].copy()
        b = inp[:, j].copy()
        inp[:, i] = np.minimum(a, b)
        inp[:, j] = np.maximum(a, b)
    return np.all(inp[:, :-1] <= inp[:, 1:])

def _odd_even_transposition():
    """Odd-even transposition sorting network for N wires."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def construct():
    global _INPUTS
    _start_time = time.time()
    DEADLINE = _start_time + 80.0  # 80s to leave 10s margin before 90s kill

    # Precompute inputs
    _INPUTS = _precompute_inputs()

    # Start with odd-even transposition
    network = _odd_even_transposition()
    
    # Verify initial network
    assert _verify_network(network), "Initial network failed verification"
    
    best_count = len(network)
    
    # Local search: try to delete comparators
    # We'll iterate: try to delete each comparator, check if still valid
    # If valid, keep deleted. Repeat until no more deletions possible.
    # Then try mutations (swap two comparators, or move one) to enable further deletions.
    
    def _try_delete(network):
        """Try to delete each comparator in reverse order.
        Returns (new_network, num_deleted)."""
        deleted = 0
        new_network = list(network)
        # Try deleting from the end (often later comparators are more redundant)
        for idx in range(len(new_network) - 1, -1, -1):
            if time.time() > DEADLINE:
                break
            # Temporarily remove comparator at idx
            candidate = new_network[:idx] + new_network[idx+1:]
            if _verify_network(candidate, _INPUTS):
                new_network = candidate
                deleted += 1
        return new_network, deleted
    
    def _try_mutate(network, max_mutations=50):
        """Try random mutations to enable further deletions.
        Returns the best network found."""
        best_net = list(network)
        best_len = len(network)
        
        for _ in range(max_mutations):
            if time.time() > DEADLINE:
                break
            candidate = list(best_net)
            # Random mutation: swap two comparators, or move one
            if len(candidate) >= 2:
                if random.random() < 0.5:
                    # Swap two comparators
                    i1 = random.randint(0, len(candidate) - 1)
                    i2 = random.randint(0, len(candidate) - 1)
                    candidate[i1], candidate[i2] = candidate[i2], candidate[i1]
                else:
                    # Move a comparator
                    i1 = random.randint(0, len(candidate) - 1)
                    i2 = random.randint(0, len(candidate) - 1)
                    comp = candidate.pop(i1)
                    candidate.insert(i2, comp)
            
            # Verify
            if _verify_network(candidate, _INPUTS):
                # Try to delete from this candidate
                candidate2, deleted = _try_delete(candidate)
                if len(candidate2) < best_len:
                    best_net = candidate2
                    best_len = len(candidate2)
        
        return best_net
    
    # Phase 1: Greedy deletion
    network, _ = _try_delete(network)
    best_count = len(network)
    
    # Phase 2: Iterative mutation + deletion
    improvement = True
    while improvement and time.time() < DEADLINE:
        improvement = False
        # Try several random restarts of mutation
        for attempt in range(10):
            if time.time() > DEADLINE:
                break
            mutated = _try_mutate(network, max_mutations=30)
            if len(mutated) < len(network):
                network = mutated
                improvement = True
                best_count = len(network)
    
    # Final verification
    assert _verify_network(network, _INPUTS), "Final network failed verification"
    
    return network

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
