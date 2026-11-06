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

def _get_sample_inputs(n_samples=8192):
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

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Uses the recursive odd-even merge construction.
    Returns list of (i, j) comparator pairs.
    """
    if n == 1:
        return []
    
    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        # Recursively sort first half and second half (offset)
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        
        # Odd-even merge
        merged = _odd_even_merge(n)
        
        return left + right + merged
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        comparators = []
        
        if n % 2 == 0:
            # Even case: compare adjacent pairs starting from 0
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            # Now recursively merge: even positions form one sub-problem, odd form another
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            # Odd case: compare (1,2),(3,4),...
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            # Element 0 stays, element n-1 stays
            # Recursively merge the middle: positions 1..n-2 (n-2 elements)
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        
        return comparators
    
    return _build(n)

def odd_even_transposition_network(n):
    """Generate odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def improve_by_removal(comparators, deadline):
    """
    Iteratively try to remove each comparator if the network remains correct.
    Uses fast validation first, then full validation.
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

def try_insert_and_remove(best, deadline, max_attempts=500):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    This can escape local minima.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    
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

def swap_comparators(comparators, deadline, max_attempts=200):
    """
    Try swapping two comparators in the network to see if it enables further removal.
    """
    best = list(comparators)
    best_len = len(best)
    
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Pick two random positions to swap
        if len(best) < 2:
            break
        p1 = random.randint(0, len(best) - 1)
        p2 = random.randint(0, len(best) - 1)
        if p1 == p2:
            continue
        
        candidate = best[:]
        candidate[p1], candidate[p2] = candidate[p2], candidate[p1]
        
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
    deadline = start_time + 82  # Safety margin before 90s kill
    
    # Generate multiple base networks
    base_merge = odd_even_merge_sort_network(N)
    base_trans = odd_even_transposition_network(N)
    
    # Verify bases are correct
    if not is_sorted_network(base_merge):
        base_merge = base_trans
    
    best = list(base_merge)
    best_len = len(best)
    
    # Phase 1: Remove redundant comparators from merge base
    if time.time() < deadline - 15:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 2: Try insert-then-remove to escape local minima
    if time.time() < deadline - 15:
        best = try_insert_and_remove(best, deadline, max_attempts=300)
        best_len = len(best)
    
    # Phase 3: Try swapping comparators
    if time.time() < deadline - 10:
        best = swap_comparators(best, deadline, max_attempts=150)
        best_len = len(best)
    
    # Phase 4: Another removal pass
    if time.time() < deadline - 8:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 5: More insert-and-remove cycles
    if time.time() < deadline - 6:
        best = try_insert_and_remove(best, deadline, max_attempts=200)
        best_len = len(best)
    
    # Phase 6: Final removal pass
    if time.time() < deadline - 3:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        # Should not happen, but fallback
        return base_trans
    
    return best
# EVOLVE-BLOCK-END
