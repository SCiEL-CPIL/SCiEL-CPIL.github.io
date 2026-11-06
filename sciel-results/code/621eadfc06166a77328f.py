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

def _try_remove_all_redundant(comparators, deadline):
    """
    Iteratively try to remove each comparator from the list.
    Returns the reduced list.
    """
    best = list(comparators)
    i = 0
    while i < len(best):
        if time.time() > deadline:
            break
        # Try removing comparator i
        candidate = best[:i] + best[i+1:]
        if is_sorted_network(candidate):
            best = candidate
            # Don't increment i, check same position again with new list
        else:
            i += 1
    return best

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
    best = _try_remove_all_redundant(best, deadline)
    best_len = len(best)
    
    # Try adding comparators at random positions to enable further removals
    # This is a simple hill-climbing step
    attempts = 0
    while time.time() < deadline and attempts < max_steps:
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
            candidate = _try_remove_all_redundant(candidate, deadline)
            if len(candidate) < len(best):
                best = candidate
                best_len = len(best)
    
    return best

def _build_bitonic_network(n_wires):
    """
    Build a bitonic sorting network on n_wires wires.
    Returns list of (i, j) comparators.
    """
    comparators = []
    
    def bitonic_merge(lo, hi):
        """Merge a bitonic sequence from lo to hi-1 (hi exclusive)."""
        if hi - lo <= 1:
            return
        # Compare elements that are (hi-lo)/2 apart
        step = (hi - lo) // 2
        for i in range(lo, hi - step):
            comparators.append((i, i + step))
        # Recursively sort the two halves
        bitonic_merge(lo, lo + step)
        bitonic_merge(lo + step, hi)
    
    def bitonic_sort(lo, hi, direction=1):
        """
        Sort a bitonic sequence from lo to hi-1.
        direction: 1 for ascending, -1 for descending
        """
        if hi - lo <= 1:
            return
        step = (hi - lo) // 2
        # First make it bitonic: sort first half ascending, second half descending
        bitonic_sort(lo, lo + step, 1)
        bitonic_sort(lo + step, hi, -1)
        # Then merge
        bitonic_merge(lo, hi)
    
    # Build the full bitonic network
    # For N wires, we need log2(N) stages of building bitonic sequences
    # Standard bitonic sort:
    # Stage k: for each block of size 2^k, sort it to be bitonic
    # Then final merge
    
    # Actually, let's use the standard construction:
    # For each k from 1 to log2(N):
    #   For each block of size 2^k:
    #     Sort the block to be bitonic (first half asc, second half desc)
    # Then final merge of the whole array
    
    # Simpler: use recursive approach
    # For N=17, we need to handle non-power-of-2
    # Pad to 32 wires? No, let's just do it properly for 17.
    
    # Actually, bitonic sort works for any N, but the standard construction
    # assumes power of 2. Let's use a different approach.
    
    # Let's build it iteratively:
    # For k = 1, 2, 4, 8, ... (powers of 2 up to N):
    #   For each block starting at 0, 2^k, 2*2^k, ...:
    #     Sort the block to be bitonic
    
    comparators = []
    k = 1
    while k < n_wires:
        # For each block of size 2*k
        for start in range(0, n_wires, 2 * k):
            end = min(start + 2 * k, n_wires)
            if end - start < 2:
                continue
            # Sort this block to be bitonic
            # First half ascending, second half descending
            mid = start + (end - start) // 2
            # Sort first half (start to mid) ascending
            _sort_ascending(comparators, start, mid)
            # Sort second half (mid to end) descending
            _sort_descending(comparators, mid, end)
        k *= 2
    
    # Final merge
    _merge_bitonic(comparators, 0, n_wires)
    
    return comparators

def _sort_ascending(comparators, lo, hi):
    """Sort wires lo to hi-1 in ascending order using bitonic sort."""
    if hi - lo <= 1:
        return
    mid = (lo + hi) // 2
    _sort_ascending(comparators, lo, mid)
    _sort_ascending(comparators, mid, hi)
    _merge_bitonic(comparators, lo, hi)

def _sort_descending(comparators, lo, hi):
    """Sort wires lo to hi-1 in descending order using bitonic sort."""
    if hi - lo <= 1:
        return
    mid = (lo + hi) // 2
    _sort_descending(comparators, lo, mid)
    _sort_descending(comparators, mid, hi)
    _merge_bitonic(comparators, lo, hi)

def _merge_bitonic(comparators, lo, hi):
    """Merge a bitonic sequence from lo to hi-1."""
    if hi - lo <= 1:
        return
    step = (hi - lo) // 2
    for i in range(lo, hi - step):
        comparators.append((i, i + step))
    _merge_bitonic(comparators, lo, lo + step)
    _merge_bitonic(comparators, lo + step, hi)

def construct():
    """
    Construct a sorting network on 17 wires with as few comparators as possible.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Start with the seed odd-even transposition network
    seed = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    
    best = list(seed)
    
    # Verify seed is correct
    if not is_sorted_network(best):
        raise ValueError("Seed network is not correct")
    
    # Try to improve by removing redundant comparators
    best = improve_network(best, max_steps=20000, deadline=deadline)
    
    # Try bitonic network as an alternative starting point
    if time.time() < deadline - 10:
        try:
            bitonic = _build_bitonic_network(N)
            if is_sorted_network(bitonic):
                bitonic_improved = improve_network(bitonic, max_steps=5000, deadline=deadline)
                if len(bitonic_improved) < len(best):
                    best = bitonic_improved
        except Exception:
            pass
    
    # Try a few random restarts with smaller networks
    random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 10:
            break
        
        # Generate a random network slightly smaller than current best
        target_len = max(60, len(best) - random.randint(5, 15))
        if target_len < len(best):
            candidate = random_network(target_len)
            if is_sorted_network(candidate):
                best = candidate
    
    # Final attempt: try to further reduce the best network
    if time.time() < deadline - 3:
        best = improve_network(best, max_steps=10000, deadline=deadline)
    
    return best

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
