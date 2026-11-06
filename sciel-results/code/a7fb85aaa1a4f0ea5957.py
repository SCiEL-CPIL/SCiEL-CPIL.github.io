"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None
_SAMPLE_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_inputs(n_samples=4096):
    global _SAMPLE_INPUTS
    if _SAMPLE_INPUTS is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
        _SAMPLE_INPUTS = all_inputs[idx]
    return _SAMPLE_INPUTS

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_fast(comparators, inputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs = _get_sample_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires.
    """
    if n == 1:
        return []
    
    def _bitonic_merge(n, offset=0, direction=1):
        # direction 1: ascending, -1: descending
        if n == 1:
            return []
        comparators = []
        # Compare pairs (offset, offset+n/2), (offset+1, offset+n/2+1)...
        half = n // 2
        for i in range(half):
            i1 = offset + i
            i2 = offset + i + half
            if direction == 1:
                if i1 > i2: i1, i2 = i2, i1
            else:
                if i1 < i2: i1, i2 = i2, i1
            comparators.append((i1, i2))
        
        # Recursively merge the two halves
        # Left half: offset .. offset+half-1
        # Right half: offset+half .. offset+n-1
        comparators.extend(_bitonic_merge(half, offset, direction))
        comparators.extend(_bitonic_merge(half, offset + half, direction))
        return comparators

    def _bitonic_sort(n, offset=0, direction=1):
        if n == 1:
            return []
        comparators = []
        half = n // 2
        # Sort left half ascending, right half descending (to form bitonic sequence)
        comparators.extend(_bitonic_sort(half, offset, 1))
        comparators.extend(_bitonic_sort(n - half, offset + half, -1))
        # Merge the bitonic sequence
        comparators.extend(_bitonic_merge(n, offset, direction))
        return comparators

    return _bitonic_sort(n)

def improve_by_removal(comparators, deadline):
    """
    Iteratively try to remove each comparator if the network remains correct.
    """
    best = list(comparators)
    
    i = 0
    while i < len(best):
        if time.time() > deadline - 3:
            break
        candidate = best[:i] + best[i+1:]
        # Fast check first
        if is_sorted_fast(candidate):
            # Full check
            if is_sorted_network(candidate):
                best = candidate
                # Don't increment i, re-check same position
            else:
                i += 1
        else:
            i += 1
    
    return best

def try_insert_and_remove(best, deadline):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    max_attempts = 500 # Increased attempts
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Insert a random comparator at a random position
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Try to remove comparators from the candidate
        k = 0
        improved = True
        while k < len(candidate) and improved:
            improved = False
            if time.time() > deadline - 3:
                break
            test = candidate[:k] + candidate[k+1:]
            if is_sorted_fast(test):
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                    # Don't increment k
                else:
                    k += 1
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # Use Bitonic Sort as the base network (smaller than transposition)
    # For 17 wires, Bitonic Sort is typically ~80-90 comparators
    base = bitonic_sort_network(N)
    
    # Verify base is correct
    if not is_sorted_network(base):
        # Fallback to odd-even transposition if bitonic fails (should not happen)
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Remove redundant comparators
    if time.time() < deadline - 10:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 2: Try insert-then-remove to escape local minima
    if time.time() < deadline - 10:
        best = try_insert_and_remove(best, deadline)
        best_len = len(best)
    
    # Phase 3: Another pass of removal
    if time.time() < deadline - 5:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 4: Final removal pass
    if time.time() < deadline - 3:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        # Should not happen, but fallback
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END
