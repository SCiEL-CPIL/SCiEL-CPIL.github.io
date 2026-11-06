"""Improved sorting network search with better initial network and optimized validation."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute masks for fast simulation
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute a smaller sample set for fast validation during search
_SAMPLE_SIZE = 8192
_rng = np.random.RandomState(42)
_SAMPLE_INDICES = _rng.choice(_NUM_INPUTS, size=_SAMPLE_SIZE, replace=False)
_SAMPLE_INPUTS = _INPUTS[_SAMPLE_INDICES]
_SAMPLE_SORTED = _SORTED_INPUTS[_SAMPLE_INDICES]


def simulate_network(comparators, inputs):
    """
    inputs: numpy array of shape (num_inputs, N) with values in {0, 1}
    comparators: list of (i, j)
    returns: sorted numpy array of shape (num_inputs, N)
    """
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return state


def is_sorted_network_full(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
    Uses precomputed input matrix for speed.
    """
    if len(comparators) > 4096:
        return False
    
    state = _INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    return np.array_equal(state, _SORTED_INPUTS)


def is_sorted_network_sample(comparators):
    """
    Fast check on a sample of inputs. Not sufficient for final validation,
    but much faster for guiding search.
    """
    if len(comparators) > 4096:
        return False
    
    state = _SAMPLE_INPUTS.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    
    return np.array_equal(state, _SAMPLE_SORTED)


def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)


def random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators


def odd_even_transposition_network(n):
    """Generate odd-even transposition network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Returns list of (i, j) comparators.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    left = odd_even_merge_sort_network(half)
    right = odd_even_merge_sort_network(n - half)
    
    # Offset right half comparators
    right = [(i + half, j + half) for i, j in right]
    
    # Merge the two halves using odd-even merge
    merge = []
    for phase in range(n - 1):
        offset = phase % 2
        for i in range(offset, n - 1, 2):
            merge.append((i, i + 1))
    
    return left + right + merge


def bitonic_sort_network_power_of_2(n):
    """
    Generate a bitonic sort network for n wires where n is a power of 2.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort first half ascending, second half descending
    left = bitonic_sort_network_power_of_2(half)
    right = bitonic_sort_network_power_of_2(half)
    # Reverse the right half by swapping all comparators
    right = [(j, i) for i, j in right]
    right = [(i + half, j + half) for i, j in right]
    
    # Merge phase: compare (i, i+half)
    merge = []
    for i in range(half):
        merge.append((i, i + half))
    
    return left + right + merge


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Start with a good initial network
    # Try odd-even merge sort first (should be better than transposition)
    seed_network = odd_even_merge_sort_network(N)
    if not is_sorted_network_full(seed_network):
        # Fallback to odd-even transposition
        seed_network = odd_even_transposition_network(N)
    
    best_network = list(seed_network)
    best_len = len(best_network)
    
    current = list(best_network)
    
    # Phase 1: Systematic deletion - try to remove each comparator if possible
    i = 0
    while i < len(current) and time.time() < deadline - 20.0:
        candidate = current.copy()
        candidate.pop(i)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                continue
        i += 1
    
    # Phase 2: Multiple passes of systematic deletion
    for _ in range(10):
        if time.time() > deadline - 15.0:
            break
        progress = False
        i = 0
        while i < len(current) and time.time() < deadline - 15.0:
            candidate = current.copy()
            candidate.pop(i)
            if is_sorted_network_sample(candidate):
                if is_sorted_network_full(candidate):
                    current = candidate
                    best_network = candidate
                    best_len = len(current)
                    progress = True
                    continue
            i += 1
        if not progress:
            break
    
    # Phase 3: Random deletion attempts
    attempts = 0
    while time.time() < deadline - 8.0 and attempts < 100:
        attempts += 1
        if not current:
            break
        idx = random.randint(0, len(current) - 1)
        candidate = current.copy()
        candidate.pop(idx)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
    
    # Phase 4: Local search - swap adjacent comparators
    if time.time() < deadline - 4.0:
        for _ in range(200):
            if time.time() > deadline - 4.0:
                break
            if len(current) < 2:
                break
            idx = random.randint(0, len(current) - 1)
            
            # Try swapping with neighbor
            if idx > 0:
                candidate = current.copy()
                candidate[idx], candidate[idx-1] = candidate[idx-1], candidate[idx]
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current = candidate
                        best_network = candidate
                        best_len = len(current)
            elif idx < len(current) - 1:
                candidate = current.copy()
                candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current = candidate
                        best_network = candidate
                        best_len = len(current)
    
    # Phase 5: Try replacing a comparator with a different one
    if time.time() < deadline - 3.0:
        for _ in range(100):
            if time.time() > deadline - 3.0:
                break
            if not current:
                break
            idx = random.randint(0, len(current) - 1)
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            if (i, j) != current[idx]:
                candidate = current.copy()
                candidate[idx] = (i, j)
                if is_sorted_network_sample(candidate):
                    if is_sorted_network_full(candidate):
                        current = candidate
                        best_network = candidate
                        best_len = len(current)
    
    # Final systematic pass
    i = 0
    while i < len(current) and time.time() < deadline - 2.0:
        candidate = current.copy()
        candidate.pop(i)
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                current = candidate
                best_network = candidate
                best_len = len(current)
                continue
        i += 1
    
    # Final Check: Ensure we return a valid network
    if not is_sorted_network_full(best_network):
        return seed_network
        
    return best_network
