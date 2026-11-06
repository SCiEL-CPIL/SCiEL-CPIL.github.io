"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute all Boolean inputs as a numpy array of shape (2^N, N)
# Each row is a bit-vector of length N
def _make_all_inputs():
    """Generate all 2^N Boolean inputs as uint8 array of shape (2^N, N)."""
    n_inputs = 1 << N
    # Use bitwise operations to generate all combinations
    # For each bit position k (0 to N-1), the bit is set in every other input
    # We want: input i has bit k = (i >> k) & 1
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def simulate_network(comparators, inputs=None):
    """
    Simulate a compare-exchange network on all Boolean inputs.
    comparators: list of (i, j) pairs
    inputs: optional numpy array; if None, uses all 2^N Boolean inputs
    Returns the output array of shape (num_inputs, N).
    """
    if inputs is None:
        inputs = _get_all_inputs()
    state = inputs.copy()
    for (i, j) in comparators:
        # min goes to i, max goes to j
        # min(a,b) = a & b, max(a,b) = a | b for bits
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Check if the network sorts all Boolean inputs using the zero-one principle."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    # Check if each row is sorted: non-decreasing from left to right
    # For binary, sorted means all 0s then all 1s
    # Check: for each row, no 1 appears before a 0
    # Equivalent: for each adjacent pair in output, output[:,k] <= output[:,k+1]
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def random_network(num_comparators):
    """Generate a random network with the given number of comparators."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def improve_network(comparators, max_steps=5000, deadline=None):
    """
    Try to improve a network by removing redundant comparators or swapping them.
    Uses local search: try removing each comparator, check if still correct.
    """
    if deadline is None:
        deadline = time.time() + 60
    
    best = list(comparators)
    best_len = len(best)
    
    # First, try to remove redundant comparators
    # Iterate through and try removing each one
    i = 0
    while i < len(best):
        if time.time() > deadline - 5:
            break
        # Try removing comparator i
        candidate = best[:i] + best[i+1:]
        if is_sorted_network(candidate):
            best = candidate
            best_len = len(best)
            # Don't increment i, check same position again with new list
        else:
            i += 1
    
    # Try adding comparators at random positions to enable further removals
    # This is a simple hill-climbing step
    attempts = 0
    while time.time() < deadline - 5 and attempts < max_steps:
        attempts += 1
        # Pick a random position to insert a new comparator
        if not best:
            pos = 0
        else:
            pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Check if still sorted
        if is_sorted_network(candidate):
            # Now try to remove some comparators
            k = 0
            improved = True
            while k < len(candidate) and improved:
                improved = False
                # Try removing comparator k
                test = candidate[:k] + candidate[k+1:]
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                    # Don't increment k
                else:
                    k += 1
            if len(candidate) < len(best):
                best = candidate
                best_len = len(best)
    
    return best

def construct():
    """
    Construct a sorting network on 17 wires with as few comparators as possible.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Start with the seed odd-even transposition network
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    best = list(seed)
    
    # Verify seed is correct
    if not is_sorted_network(best):
        raise ValueError("Seed network is not correct")
    
    # Try to improve by removing redundant comparators
    best = improve_network(best, deadline=deadline)
    
    # Try a few random restarts with smaller networks
    random.seed(42)
    for restart in range(10):
        if time.time() > deadline - 10:
            break
        
        # Generate a random network slightly smaller than current best
        target_len = max(60, len(best) - random.randint(5, 20))
        if target_len < len(best):
            candidate = random_network(target_len)
            # Try to fix it by adding necessary comparators
            # Use a heuristic: start from a known good structure
            # For now, just check if it works (unlikely for random)
            if is_sorted_network(candidate):
                best = candidate
    
    # Final attempt: try to further reduce the best network
    if time.time() < deadline - 5:
        best = improve_network(best, deadline=deadline)
    
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
