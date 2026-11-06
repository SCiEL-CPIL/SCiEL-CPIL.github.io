"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
MAX_COMPARATORS = 4096
DEADLINE = 85.0  # safety margin before 90s wall-clock kill

# Precompute all 131,072 Boolean inputs as a (2^N, N) uint8 matrix
# and a target sorted matrix (all zeros then all ones)
def _build_test_data():
    """Build the 131072 x 17 test matrix and the sorted reference."""
    n_inputs = 1 << N
    # Use bit manipulation to create all binary vectors
    # For efficiency, we generate them in a structured way
    # Row k represents binary of k, but we need consistent bit ordering
    # Let's use: input[i] = (k >> i) & 1 for i in range(N)
    # This gives us all 2^17 distinct inputs
    
    # We'll store as uint8 array for memory efficiency
    # 131072 * 17 = 2,228,224 bytes ~ 2.2 MB - fine
    
    inputs = np.zeros((n_inputs, N), dtype=np.uint8)
    for i in range(N):
        # Column i: bit i of each row index
        # We can use broadcasting
        col = (np.arange(n_inputs, dtype=np.uint32) >> i) & 1
        inputs[:, i] = col
    
    # Target: sorted means wire 0 has min (0), wire 16 has max
    # For a binary input with k ones: sorted output has (N-k) zeros then k ones
    # So target[j] = 1 if j >= N - k, i.e., target[j] = 1 if j >= N - popcount(row)
    # Equivalently: target[j] = 1 iff the number of ones > j (0-indexed)
    # i.e., popcount(row) > j
    
    # Compute popcount for each row
    popcounts = np.zeros(n_inputs, dtype=np.uint8)
    for i in range(N):
        popcounts += inputs[:, i]
    
    # Target: target[row, j] = 1 if popcounts[row] > j
    # i.e., target[row, j] = 1 if j < popcounts[row]
    # Wait: sorted ascending means smallest on wire 0.
    # For binary: sorted ascending = [0,0,...,0,1,1,...,1]
    # Number of zeros = N - popcount, number of ones = popcount
    # So target[row, j] = 0 if j < N - popcounts[row], else 1
    # i.e., target[row, j] = 1 if j >= N - popcounts[row]
    # i.e., target[row, j] = 1 if j > N - 1 - popcounts[row]
    # i.e., target[row, j] = 1 if N - 1 - j < popcounts[row]
    # i.e., target[row, j] = 1 if popcounts[row] > N - 1 - j
    
    target = np.zeros((n_inputs, N), dtype=np.uint8)
    for j in range(N):
        threshold = N - 1 - j  # need popcount > threshold
        target[:, j] = (popcounts > threshold).astype(np.uint8)
    
    return inputs, target, popcounts

# Global test data
_inputs, _target, _popcounts = _build_test_data()


def apply_network(comparators, data):
    """Apply a list of comparators to the input data matrix.
    Returns the output matrix."""
    result = data.copy()
    for i, j in comparators:
        # Compare-exchange: min to wire i, max to wire j
        ri = result[:, i]
        rj = result[:, j]
        # min = min(ri, rj), max = max(ri, rj)
        # For binary: min = ri & rj, max = ri | rj
        # But let's use general min/max for safety
        result[:, i] = np.minimum(ri, rj)
        result[:, j] = np.maximum(ri, rj)
    return result


def is_sorted_network(comparators):
    """Check if the comparator network sorts all Boolean inputs."""
    output = apply_network(comparators, _inputs)
    return np.array_equal(output, _target)


def is_sorted_network_fast(comparators):
    """Faster check using in-place operations and early termination."""
    # Work on a copy
    result = _inputs.copy()
    for i, j in comparators:
        ri = result[:, i]
        rj = result[:, j]
        # For binary data: min = AND, max = OR
        # But we need to be careful: after first comparator, values are still 0/1
        # So AND/OR work correctly
        result[:, i] = ri & rj
        result[:, j] = ri | rj
    
    # Check against target
    # Instead of full array_equal, we can check row by row or use sum
    # Actually, let's just do the comparison
    return np.array_equal(result, _target)


def network_depth(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    # Wire depths
    depths = [0] * N
    for i, j in comparators:
        # The comparator (i,j) can only execute after both wires i and j have finished
        # their previous operations
        d = max(depths[i], depths[j]) + 1
        depths[i] = d
        depths[j] = d
    return max(depths)


def odd_even_transposition_network():
    """Generate the odd-even transposition sorting network."""
    comps = []
    for phase in range(N):
        if phase % 2 == 0:
            for i in range(0, N - 1, 2):
                comps.append((i, i + 1))
        else:
            for i in range(1, N - 1, 2):
                comps.append((i, i + 1))
    return comps


def bubble_sort_network():
    """Generate a simple bubble sort network."""
    comps = []
    for i in range(N):
        for j in range(N - 1 - i):
            comps.append((j, j + 1))
    return comps


def bitonic_sort_network():
    """Generate a bitonic sort network for 17 wires (pad to 32? No, let's do a custom one).
    Actually, bitonic sort works for powers of 2. For 17, we'll use a different approach.
    Let's use a simple recursive odd-even merge based network or just use the transposition."""
    # For non-power-of-2, let's just use odd-even transposition as base
    return odd_even_transposition_network()


def try_delete_comparators(comparators, time_budget=30.0):
    """Try to remove unnecessary comparators using systematic deletion.
    Returns the minimized list of comparators."""
    start_time = time.time()
    current = list(comparators)
    
    # First verify it's sorted
    if not is_sorted_network_fast(current):
        return current
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        i = 0
        while i < len(current):
            if time.time() - start_time > time_budget:
                break
            # Try removing comparator at position i
            candidate = current[:i] + current[i+1:]
            if is_sorted_network_fast(candidate):
                current = candidate
                improved = True
                # Don't increment i, check same position again
            else:
                i += 1
    
    return current


def random_search(time_budget=20.0):
    """Random search for better networks.
    Strategy: start from a valid network, try random modifications."""
    start_time = time.time()
    
    # Start from odd-even transposition
    base = odd_even_transposition_network()
    current = list(base)
    
    # First minimize
    current = try_delete_comparators(current, time_budget=20.0)
    
    if time.time() - start_time > time_budget:
        return current
    
    # Now try to improve further with random insertions/deletions
    best = list(current)
    best_count = len(best)
    
    # Define all possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    rng = random.Random(42)
    
    while time.time() - start_time < time_budget:
        # Random operation: delete a random comparator, or swap, or insert
        op = rng.random()
        
        if op < 0.5 and len(current) > 10:
            # Try deleting a random comparator
            idx = rng.randint(0, len(current) - 1)
            candidate = current[:idx] + current[idx+1:]
            if is_sorted_network_fast(candidate):
                current = candidate
                if len(current) < best_count:
                    best = list(current)
                    best_count = len(current)
        elif op < 0.8:
            # Try swapping two comparators
            if len(current) >= 2:
                i, j = rng.sample(range(len(current)), 2)
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if is_sorted_network_fast(candidate):
                    current = candidate
                    if len(current) < best_count:
                        best = list(current)
                        best_count = len(current)
        else:
            # Try inserting a random comparator at a random position
            if len(current) < MAX_COMPARATORS:
                comp = rng.choice(all_comps)
                pos = rng.randint(0, len(current))
                candidate = current[:pos] + [comp] + current[pos:]
                if is_sorted_network_fast(candidate):
                    current = candidate
                    # After insertion, try to delete
                    current = try_delete_comparators(current, time_budget=5.0)
                    if len(current) < best_count:
                        best = list(current)
                        best_count = len(current)
    
    return best


def systematic_construction():
    """Try to build a good network from scratch using a greedy approach.
    This is a heuristic: repeatedly add the comparator that most helps sorting."""
    start_time = time.time()
    
    # Start with empty network
    current = []
    
    # We need to add comparators until the network sorts all inputs
    # Greedy: at each step, add the comparator that reduces the "distance" to sorted
    
    # Precompute: for each input, what's the current state
    # This is expensive, so let's use a different strategy
    
    # Actually, let's just use the odd-even transposition and minimize
    base = odd_even_transposition_network()
    return try_delete_comparators(base, time_budget=40.0)


def construct():
    """Construct a compare-exchange network that sorts all 17-input sequences."""
    start_time = time.time()
    
    # Strategy 1: Start with odd-even transposition and minimize
    base = odd_even_transposition_network()
    result = try_delete_comparators(base, time_budget=30.0)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return result
    
    # Strategy 2: Try random search for improvements
    result2 = random_search(time_budget=DEADLINE - elapsed - 5)
    
    # Return the best result
    if len(result2) < len(result):
        result = result2
    
    return result


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
