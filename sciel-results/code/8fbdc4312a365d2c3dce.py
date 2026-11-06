"""
Size-efficient sorting network for 16 wires.

Strategy:
1. Generate a correct baseline (Batcher bitonic sort, 72 comparators).
2. Use greedy removal to strip redundant comparators.
3. Use local search (swap, insert, delete) to further minimize comparator count.
4. Time-boxed search with deadline guard.

The zero-one principle guarantees that if the network sorts all 2^16 boolean inputs,
it sorts all 16-input sequences.
"""
import time
import numpy as np
import random

N = 16
MAX_COMPARATORS = 4096
# Deadline guard: evaluator kills at 90s; use 80s as hard limit with 5s margin
DEADLINE = 80.0

# Pre-compute the 65536 x 16 boolean input matrix
_N_INPUTS = 1 << N
_INDICES = np.arange(_N_INPUTS, dtype=np.uint32)
_INPUT_MATRIX = np.zeros((_N_INPUTS, N), dtype=np.uint8)
for _i in range(N):
    _INPUT_MATRIX[:, _i] = (_INDICES >> _i) & 1

# Pre-compute sorted check: for each row, is it sorted?
# A row is sorted iff for all i in 0..N-2, x[i] <= x[i+1]
# We can precompute a "sorted mask" function quickly.


def _is_sorted_batch(x):
    """Check if all rows in x (shape (M, N)) are sorted. Returns bool."""
    # For each adjacent pair, check x[:, i] > x[:, i+1]
    for i in range(N - 1):
        if np.any(x[:, i] > x[:, i + 1]):
            return False
    return True


def _apply_network_fast(network, x):
    """
    Apply network to pre-allocated array x (in-place).
    network: list of (i, j) pairs.
    x: numpy array of shape (65536, 16), uint8.
    """
    for i, j in network:
        mask = x[:, i] > x[:, j]
        if np.any(mask):
            x[mask, i] = 0
            x[mask, j] = 1


def _verify_network(network):
    """
    Verify that the network sorts all boolean inputs.
    Returns True if valid, False otherwise.
    """
    if not network:
        return N == 1
    
    # Check for duplicate comparators (not strictly invalid but wasteful)
    # More importantly, check validity
    
    x = _INPUT_MATRIX.copy()
    _apply_network_fast(network, x)
    return _is_sorted_batch(x)


def _verify_network_quick(network, subset_size=4096):
    """
    Quick verification on a random subset of inputs for speed during search.
    Returns True if all subset inputs are sorted.
    """
    if not network:
        return N == 1
    
    # Use a deterministic subset for reproducibility
    idx = np.random.choice(_N_INPUTS, size=subset_size, replace=False)
    x = _INPUT_MATRIX[idx].copy()
    _apply_network_fast(network, x)
    return _is_sorted_batch(x)


def _generate_bitonic_sort(n):
    """
    Generate Batcher's bitonic sort network for n wires (power of 2).
    """
    def bitonic_merge(n, offset=0, ascending=True):
        if n == 1:
            return []
        half = n // 2
        comparators = []
        for i in range(half):
            a = i + offset
            b = i + half + offset
            if not ascending:
                a, b = b, a
            comparators.append((a, b))
        comparators.extend(bitonic_merge(half, offset, ascending))
        comparators.extend(bitonic_merge(half, half + offset, not ascending))
        return comparators
    
    def bitonic_sort(n, offset=0, ascending=True):
        if n == 1:
            return []
        half = n // 2
        comparators = []
        comparators.extend(bitonic_sort(half, offset, True))
        comparators.extend(bitonic_sort(half, half + offset, False))
        comparators.extend(bitonic_merge(n, offset, ascending))
        return comparators
    
    return bitonic_sort(n)


def _greedy_removal(network, deadline):
    """
    Greedily remove comparators that are not needed.
    Iterates through the network, tries removing each one, and keeps it
    if removal breaks correctness.
    """
    current = list(network)
    start_time = time.time()
    
    # Repeated passes until no more removals or deadline
    improved = True
    while improved and time.time() < deadline:
        improved = False
        i = 0
        while i < len(current):
            if time.time() >= deadline:
                break
            
            # Try removing comparator at position i
            trial = current[:i] + current[i+1:]
            
            if _verify_network(trial):
                current = trial
                improved = True
                # Don't increment i, check same position again
            else:
                i += 1
    
    return current


def _local_search(network, deadline, max_iter=5000):
    """
    Local search with swap, insert, and delete moves.
    """
    current = list(network)
    best_count = len(current)
    start_time = time.time()
    random.seed(42)
    
    for iteration in range(max_iter):
        if time.time() >= deadline:
            break
        
        # Choose a random move type
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap two comparators
            if len(current) < 2:
                continue
            i, j = random.sample(range(len(current)), 2)
            trial = current[:]
            trial[i], trial[j] = trial[j], trial[i]
        elif move_type < 0.7:
            # Insert a random comparator
            a, b = random.sample(range(N), 2)
            if a > b:
                a, b = b, a
            pos = random.randint(0, len(current))
            trial = current[:]
            trial.insert(pos, (a, b))
        else:
            # Delete a random comparator
            if not current:
                continue
            pos = random.randint(0, len(current) - 1)
            trial = current[:]
            trial.pop(pos)
        
        # Quick verification first
        if not _verify_network_quick(trial, subset_size=2048):
            continue
        
        # Full verification
        if _verify_network(trial):
            current = trial
            if len(current) < best_count:
                best_count = len(current)
    
    return current


def construct():
    """
    Construct a sorting network for 16 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    
    # Generate baseline bitonic sort network
    network = _generate_bitonic_sort(N)
    
    # Verify baseline
    if not _verify_network(network):
        # Fallback to odd-even transposition
        network = [(i, i + 1) for phase in range(N)
                   for i in range(phase % 2, N - 1, 2)]
    
    # Greedy removal
    network = _greedy_removal(network, deadline=start_time + 40)
    
    # Local search
    network = _local_search(network, deadline=start_time + 75)
    
    # Final verification
    if not _verify_network(network):
        # Fallback to verified bitonic sort
        network = _generate_bitonic_sort(N)
    
    return network


# Pre-compute at module level to avoid repeated work
# The _INPUT_MATRIX is already computed above.
