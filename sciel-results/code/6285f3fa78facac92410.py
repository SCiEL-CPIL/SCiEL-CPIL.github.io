"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
# Deadline guard: the evaluator kills after 90s; keep a safety margin.
_DEADLINE = 75.0  # seconds
_START_TIME = time.time()

def _elapsed():
    return time.time() - _START_TIME

# EVOLVE-BLOCK-START

def _apply_comparator(state, i, j):
    """Apply a compare-exchange to the state array (min to i, max to j)."""
    a, b = state[i], state[j]
    if a > b:
        state[i], state[j] = b, a

def _is_sorted(state):
    """Check if the state is sorted in non-decreasing order."""
    return all(state[i] <= state[i+1] for i in range(N-1))

def _verify_network(comparators):
    """
    Verify the sorting network using the zero-one principle.
    Returns True if the network sorts all Boolean inputs of length N.
    """
    # There are 2^N = 131072 Boolean inputs for N=17.
    # We can vectorize this using numpy for speed.
    # Represent each Boolean input as a bitmask in a 131072 x N array? No, that's too big.
    # Instead, iterate over all 2^N inputs, but do it efficiently.
    
    # Actually, 131072 inputs * 71 comparators * 17 wires is manageable if optimized.
    # Let's use a bit-parallel approach or just loop.
    
    # For N=17, 2^17 = 131072 inputs.
    # We'll iterate through all of them.
    
    for mask in range(1 << N):
        state = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        for i, j in comparators:
            a, b = state[i], state[j]
            if a > b:
                state[i] = b
                state[j] = a
        if not _is_sorted(state):
            return False
    return True

def _verify_network_fast(comparators):
    """
    Faster verification using bit-parallel processing.
    Process all 2^N inputs at once using numpy.
    """
    # Each input is a bitmask from 0 to 2^N - 1.
    # We represent the state of all inputs as an array of shape (2^N, N).
    # But 131072 * 17 = 2,228,224 integers, which is fine.
    
    # Create all input vectors
    # For each bit position i, the bit is set for inputs where (mask >> i) & 1 == 1
    # This creates a pattern: 0,0,1,1,0,0,1,1,... with period 2^(i+1)
    
    n_inputs = 1 << N
    # Build state array of shape (n_inputs, N)
    # state[k, i] = (k >> i) & 1
    
    # We can build this efficiently:
    # For wire i, the pattern is: 2^i zeros, 2^i ones, repeated
    state = np.zeros((n_inputs, N), dtype=np.int8)
    for i in range(N):
        # Pattern for wire i
        period = 1 << (i + 1)
        half = 1 << i
        # Create one period: [0]*half + [1]*half
        pattern = np.zeros(period, dtype=np.int8)
        pattern[half:] = 1
        # Tile to fill n_inputs
        n_tiles = n_inputs // period
        state[:, i] = np.tile(pattern, n_tiles)
    
    # Apply comparators
    for i, j in comparators:
        a = state[:, i]
        b = state[:, j]
        # min and max
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    # Check if sorted: state[:, i] <= state[:, i+1] for all i
    # This is equivalent to: no adjacent pair has state[:, i] > state[:, i+1]
    for i in range(N - 1):
        if np.any(state[:, i] > state[:, i + 1]):
            return False
    return True

def _count_comparators(comparators):
    return len(comparators)

def _depth_of_network(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    # Depth is the length of the longest chain of dependent comparators
    # Two comparators are dependent if they share a wire
    # We can compute this by simulating levels
    # A simpler approach: build a DAG where each comparator node points to previous ones it depends on
    # But for a small network, we can just simulate layers.
    
    # Alternative: the depth is the maximum number of comparators that can be in a sequence
    # where each consecutive pair shares a wire.
    # This is equivalent to the length of the longest path in the dependency graph.
    
    # For simplicity and speed, let's compute it differently:
    # Process comparators in order. Track the "last layer" each wire was used in.
    # When a comparator (i,j) is processed, its depth is max(last_depth[i], last_depth[j]) + 1
    # Update last_depth[i] and last_depth[j] to this depth.
    
    last_depth = [0] * N
    max_depth = 0
    for i, j in comparators:
        d = max(last_depth[i], last_depth[j]) + 1
        last_depth[i] = d
        last_depth[j] = d
        if d > max_depth:
            max_depth = d
    return max_depth

def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _akl_network():
    """
    Try to construct a smaller network using known constructions.
    For N=17, the best known is 71 comparators.
    We can try to build one using a recursive construction or known tables.
    
    Actually, let's use a different approach: start with the odd-even transposition
    and try to reduce it via local search.
    """
    return _odd_even_transposition()

def _greedy_removal(comparators, max_time=30.0):
    """
    Try to remove comparators one by one if the network remains valid.
    """
    comparators = list(comparators)
    if not _verify_network_fast(comparators):
        return comparators
    
    # Try to remove each comparator
    i = 0
    while i < len(comparators) and _elapsed() < max_time:
        # Try to remove comparator at position i
        candidate = comparators[:i] + comparators[i+1:]
        if _verify_network_fast(candidate):
            comparators = candidate
            # Don't increment i, check the same position again
        else:
            i += 1
    return comparators

def _local_search(comparators, max_time=40.0):
    """
    Perform local search: try swapping adjacent comparators, or replacing a comparator
    with a different one, to reduce the count.
    """
    comparators = list(comparators)
    if not _verify_network_fast(comparators):
        return comparators
    
    best = comparators
    best_len = len(best)
    
    # Try to find a smaller network by random restarts and local moves
    rng = random.Random(42)
    
    while _elapsed() < max_time:
        # Try random swaps of adjacent comparators
        if len(comparators) < 2:
            break
        
        # Pick a random position
        pos = rng.randint(0, len(comparators) - 2)
        candidate = comparators[:]
        candidate[pos], candidate[pos+1] = candidate[pos+1], candidate[pos]
        
        if _verify_network_fast(candidate):
            comparators = candidate
        
        # Occasionally try to remove a random comparator
        if rng.random() < 0.1 and len(comparators) > 1:
            pos = rng.randint(0, len(comparators) - 1)
            candidate = comparators[:pos] + comparators[pos+1:]
            if _verify_network_fast(candidate):
                comparators = candidate
                if len(comparators) < best_len:
                    best = comparators
                    best_len = len(best)
    
    return best if len(best) <= len(comparators) else comparators

def _random_replacement_search(comparators, max_time=35.0):
    """
    Try to replace comparators with random valid ones to potentially enable further reductions.
    """
    comparators = list(comparators)
    if not _verify_network_fast(comparators):
        return comparators
    
    rng = random.Random(123)
    best = comparators
    best_len = len(best)
    
    while _elapsed() < max_time:
        # Pick a random position and try a random replacement
        if len(comparators) < 1:
            break
        
        pos = rng.randint(0, len(comparators) - 1)
        # Generate a random comparator
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        
        candidate = comparators[:]
        candidate[pos] = (i, j)
        
        if _verify_network_fast(candidate):
            comparators = candidate
            # Try to remove some comparators after the replacement
            for k in range(len(comparators) - 1, -1, -1):
                if _elapsed() >= max_time:
                    break
                candidate2 = comparators[:k] + comparators[k+1:]
                if _verify_network_fast(candidate2):
                    comparators = candidate2
                    if len(comparators) < best_len:
                        best = comparators
                        best_len = len(best)
                    break
    
    return best

def construct():
    """
    Construct a sorting network for N=17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    
    # Start with the odd-even transposition network (136 comparators for N=17)
    # Actually, for N=17, odd-even transposition has N*N = 289? No.
    # Odd-even transposition: N phases, each with (N-1)/2 or N/2 comparators.
    # For N=17: phases 0,2,4,...,16 have 8 comparators (i=0,2,...,14 -> 8 pairs)
    # Phases 1,3,...,15 have 8 comparators (i=1,3,...,15 -> 8 pairs)
    # Total: 17 * 8 = 136 comparators. Yes.
    
    network = _odd_even_transposition()
    print(f"Initial network: {len(network)} comparators, depth: {_depth_of_network(network)}")
    
    # Verify initial network
    if not _verify_network_fast(network):
        raise ValueError("Initial network is not a valid sorting network")
    
    # Phase 1: Greedy removal
    t1 = _elapsed()
    network = _greedy_removal(network, max_time=min(20.0, _DEADLINE - t1 - 10.0))
    print(f"After greedy removal: {len(network)} comparators, depth: {_depth_of_network(network)}")
    
    # Phase 2: Local search
    t2 = _elapsed()
    remaining = _DEADLINE - t2 - 5.0
    if remaining > 5.0:
        network = _local_search(network, max_time=remaining)
        print(f"After local search: {len(network)} comparators, depth: {_depth_of_network(network)}")
    
    # Final verification
    if not _verify_network_fast(network):
        # Fall back to the original if something went wrong
        network = _odd_even_transposition()
    
    return network

# EVOLVE-BLOCK-END
