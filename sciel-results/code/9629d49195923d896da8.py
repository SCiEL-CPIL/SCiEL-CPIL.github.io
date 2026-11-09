"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
N = 18

# EVOLVE-BLOCK-START
def _bitonic_merge(comparators, lo, hi, direction):
    """Merge two sorted bitonic halves. Returns list of (i,j) comparators."""
    if hi - lo <= 1:
        return []
    mid = (lo + hi) // 2
    result = []
    for i in range(hi - lo):
        if i < mid:
            a = lo + i
            b = mid + i
            if direction == 'up':
                result.append((a, b))
            else:
                result.append((b, a))
    result += _bitonic_merge(comparators, lo, mid, 'up')
    result += _bitonic_merge(comparators, mid, hi, 'down')
    return result

def _bitonic_sort(comparators, lo, hi, direction):
    """Sort a bitonic sequence. Returns list of (i,j) comparators."""
    if hi - lo <= 1:
        return []
    mid = (lo + hi) // 2
    result = []
    result += _bitonic_sort(comparators, lo, mid, 'up')
    result += _bitonic_sort(comparators, mid, hi, 'down')
    result += _bitonic_merge(comparators, lo, hi, direction)
    return result

def _odd_even_merge(comparators, lo, hi):
    """Odd-even merge for sorted inputs. Returns list of (i,j) comparators."""
    if hi - lo <= 1:
        return []
    mid = (lo + hi) // 2
    result = []
    # Compare elements at even positions with elements at odd positions shifted by 1
    for i in range(0, hi - lo, 2):
        a = lo + i
        b = lo + i + 1
        result.append((a, b))
    # Recursively merge
    result += _odd_even_merge(comparators, lo, mid)
    result += _odd_even_merge(comparators, mid, hi)
    # Final merge
    for i in range(1, hi - lo - 1, 2):
        a = lo + i
        b = lo + i + 1
        result.append((a, b))
    return result

def _construct_bitonic_network():
    """Construct a bitonic sorting network for N wires."""
    # Pad to next power of 2
    n_padded = 1
    while n_padded < N:
        n_padded *= 2
    
    comparators = []
    # Build bitonic network for n_padded wires
    # Phase 1: Create bitonic sequences
    for size in range(2, n_padded + 1, 2):
        for start in range(0, n_padded, size):
            mid = start + size // 2
            # Create bitonic sequence by sorting first half up, second half down
            comparators += _bitonic_sort(comparators, start, mid, 'up')
            comparators += _bitonic_sort(comparators, mid, start + size, 'down')
            # Merge
            comparators += _bitonic_merge(comparators, start, start + size, 'up')
    
    # Filter to only wires < N
    comparators = [(i, j) for (i, j) in comparators if i < N and j < N]
    return comparators

def _verify_network(comparators, max_tests=10000):
    """Verify that the network sorts all inputs (using Boolean inputs for speed)."""
    # Test with random Boolean inputs
    for _ in range(max_tests):
        input_seq = [random.randint(0, 1) for _ in range(N)]
        state = input_seq[:]
        for i, j in comparators:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        # Check if sorted
        for k in range(N - 1):
            if state[k] > state[k + 1]:
                return False
    return True

def _prune_comparators(comparators, deadline):
    """Try to prune redundant comparators from the network."""
    if len(comparators) == 0:
        return comparators
    
    best = comparators[:]
    n = len(comparators)
    
    # Try removing each comparator and check if still valid
    for idx in range(n - 1, -1, -1):
        if time.time() > deadline:
            break
        
        # Try removing comparator at idx
        candidate = best[:idx] + best[idx+1:]
        
        # Quick verification with fewer tests
        if _verify_network(candidate, max_tests=5000):
            best = candidate
    
    return best

def construct():
    # Deadline guard: must finish within 90s wall-clock
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Start with bitonic network
    comparators = _construct_bitonic_network()
    
    # Verify it works
    if not _verify_network(comparators):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Try to prune redundant comparators
    if time.time() < deadline:
        comparators = _prune_comparators(comparators, deadline)
    
    # Final verification
    if not _verify_network(comparators, max_tests=20000):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return comparators
# EVOLVE-BLOCK-END
