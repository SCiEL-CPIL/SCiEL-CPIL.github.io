"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from functools import lru_cache

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(comparators):
    """Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs."""
    all_inputs = np.arange(65536, dtype=np.uint8)
    bits = ((all_inputs[:, None] >> np.arange(16)[None, :]) & 1).astype(np.int8)
    for (i, j) in comparators:
        mn = np.minimum(bits[:, i], bits[:, j])
        mx = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = mn
        bits[:, j] = mx
    diffs = np.diff(bits, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _batcher_oes(n):
    """Batcher odd-even merge sort for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher_oes(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(n)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge for n wires (power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    odd = _odd_even_merge(half)
    even = _odd_even_merge(half)
    odd_shifted = [(i + 1, j + 1) for (i, j) in odd]
    cross = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd + even + odd_shifted + cross

def _build_butterfly_network():
    """Build a butterfly-based sorting network for 16 wires.
    
    Uses butterfly layers with additional comparators to ensure sorting.
    The butterfly has log2(16)=4 stages, each with 8 comparators = 32 comparators.
    We add extra layers to ensure full sorting.
    """
    comparators = []
    n = 16
    logn = 4
    
    # Butterfly layers
    for stage in range(logn):
        for i in range(n):
            j = i ^ (1 << stage)
            if i < j:
                comparators.append((i, j))
    
    # Add adjacent comparators in multiple passes to ensure sorting
    # This is similar to odd-even transposition but interleaved with butterfly
    for pass_num in range(4):
        for i in range(0, n - 1, 2):
            comparators.append((i, i + 1))
        for i in range(1, n - 1, 2):
            comparators.append((i, i + 1))
    
    # Additional butterfly layers for robustness
    for stage in range(logn):
        for i in range(n):
            j = i ^ (1 << stage)
            if i < j:
                comparators.append((i, j))
    
    # Final adjacent passes
    for pass_num in range(2):
        for i in range(0, n - 1, 2):
            comparators.append((i, i + 1))
        for i in range(1, n - 1, 2):
            comparators.append((i, i + 1))
    
    return comparators

def _apply_comparator(state, i, j):
    """Apply a single comparator to a state (tuple of 0s and 1s).
    Returns new state with min at position i and max at position j."""
    state = list(state)
    a, b = state[i], state[j]
    if a > b:
        state[i], state[j] = b, a
    return tuple(state)

def _is_sorted_state(state):
    """Check if a state is sorted (non-decreasing)."""
    for k in range(len(state) - 1):
        if state[k] > state[k+1]:
            return False
    return True

def _canonicalize_state(state):
    """Compute canonical representative of a state under wire permutation symmetry.
    For zero-one states, the canonical form is determined by the count of 1s.
    Two states with the same number of 1s are equivalent under wire permutation."""
    ones = sum(state)
    # The canonical form for a state with k ones is: k zeros followed by k ones
    # Actually, we need to be more careful. Under permutation, any arrangement of
    # k ones and (16-k) zeros is equivalent. So the canonical representative
    # is determined solely by k.
    return ones

def _dp_sorting_network(time_budget=40.0):
    """Dynamic programming approach to find minimal sorting network.
    
    Uses the zero-one principle: we only need to consider 2^16 boolean inputs.
    By symmetry, states with the same number of 1s are equivalent under
    wire permutation. However, the sorting network structure breaks this
    symmetry, so we need to track the actual state.
    
    Instead, we use a different approach: start from the known minimal
    structure and use DP to verify and minimize.
    """
    deadline = time.time() + time_budget
    
    # For 16 wires, the known optimal is 60 comparators.
    # We'll use a constructive approach based on Batcher's algorithm
    # and then minimize.
    
    # Start with Batcher OES
    base = _batcher_oes(16)
    
    if not _verify_network_fast(base):
        return base
    
    # Use systematic removal to minimize
    result = _systematic_removal(base, time_budget=min(35.0, deadline - time.time() - 5.0))
    
    return result

def _systematic_removal(network, time_budget=50.0):
    """Systematically try to remove each comparator one at a time.
    This is more effective than random removal for finding minimal networks.
    Uses a smarter verification that tracks which inputs are still unsorted.
    """
    deadline = time.time() + time_budget
    current = list(network)
    
    if not _verify_network_fast(current):
        return current
    
    # Precompute all inputs
    all_inputs = np.arange(65536, dtype=np.uint8)
    
    def _get_unsorted_indices(comps):
        """Return set of input indices that are still unsorted after applying comps."""
        bits = ((all_inputs[:, None] >> np.arange(16)[None, :]) & 1).astype(np.int8)
        for (i, j) in comps:
            mn = np.minimum(bits[:, i], bits[:, j])
            mx = np.maximum(bits[:, i], bits[:, j])
            bits[:, i] = mn
            bits[:, j] = mx
        diffs = np.diff(bits, axis=1)
        unsorted = np.any(diffs < 0, axis=1)
        return set(np.where(unsorted)[0].tolist())
    
    # Track which inputs are still problematic
    unsorted = _get_unsorted_indices(current)
    if not unsorted:
        return current
    
    # Keep trying to remove comparators until we can't anymore
    progress = True
    while progress and time.time() < deadline - 5.0:
        progress = False
        # Try removing each comparator in order, prioritizing those that
        # don't fix any currently-unsorted input
        for idx in range(len(current)):
            if time.time() >= deadline - 5.0:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                unsorted = _get_unsorted_indices(current)
                progress = True
                break  # Restart from beginning after successful removal
    
    return current

def _greedy_minimize(network, time_budget=50.0):
    """Greedy minimization: repeatedly find and remove any removable comparator.
    Uses a smarter ordering to find removable comparators faster.
    """
    deadline = time.time() + time_budget
    current = list(network)
    
    if not _verify_network_fast(current):
        return current
    
    # Try to remove comparators, prioritizing those that appear less critical
    # First, count how many times each wire pair appears
    pair_count = {}
    for (i, j) in current:
        pair_count[(i, j)] = pair_count.get((i, j), 0) + 1
    
    # Sort indices by pair frequency (try removing frequent ones first? or rare ones?)
    # Try removing each comparator, keeping the one that allows removal
    removed_any = True
    while removed_any and time.time() < deadline - 5.0:
        removed_any = False
        # Try each position
        for idx in range(len(current)):
            if time.time() >= deadline - 5.0:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                removed_any = True
                break
    
    return current

def _construct_batcher_16():
    """Construct a known-good Batcher odd-even merge sort network for 16 wires.
    This is a well-known construction with 63 comparators."""
    return _batcher_oes(16)

def _try_remove_swap(network, time_budget=20.0):
    """Try to improve network by removing comparators and swapping adjacent ones."""
    deadline = time.time() + time_budget
    current = list(network)
    rng = random.Random(123)
    
    if not _verify_network_fast(current):
        return current
    
    # First pass: systematic removal
    current = _systematic_removal(current, time_budget=min(15.0, deadline - time.time() - 5.0))
    
    # Second pass: try swapping adjacent comparators to enable further removal
    progress = True
    while progress and time.time() < deadline - 5.0:
        progress = False
        for idx in range(len(current) - 1):
            if time.time() >= deadline - 5.0:
                break
            # Try swapping
            candidate = list(current)
            candidate[idx], candidate[idx+1] = candidate[idx+1], candidate[idx]
            # After swap, try to remove one of them
            for remove_idx in [idx, idx+1]:
                candidate2 = candidate[:remove_idx] + candidate[remove_idx+1:]
                if _verify_network_fast(candidate2):
                    current = candidate2
                    progress = True
                    break
            if progress:
                break
    
    return current

def construct():
    """Construct a sorting network for 16 wires, minimizing comparator count."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Start with Batcher OES (known correct, ~63 comparators)
    base = _batcher_oes(16)
    
    if not _verify_network_fast(base):
        # Fallback to odd-even transposition (120 comparators, definitely correct)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = base
    best_count = len(base)
    
    # Strategy 1: Systematic removal on Batcher
    if time.time() < deadline - 20.0:
        result = _systematic_removal(base, time_budget=min(25.0, deadline - time.time() - 15.0))
        if _verify_network_fast(result) and len(result) < best_count:
            best = result
            best_count = len(result)
    
    # Strategy 2: Try remove + swap optimization
    if time.time() < deadline - 15.0 and best_count > 58:
        result = _try_remove_swap(best, time_budget=min(15.0, deadline - time.time() - 10.0))
        if _verify_network_fast(result) and len(result) < best_count:
            best = result
            best_count = len(result)
    
    # Strategy 3: Local search with random restarts
    if time.time() < deadline - 10.0 and best_count > 56:
        rng = random.Random(42)
        current = list(best)
        while time.time() < deadline - 5.0:
            # Try random removal
            if len(current) > 50:
                idx = rng.randrange(len(current))
                candidate = current[:idx] + current[idx+1:]
                if _verify_network_fast(candidate):
                    current = candidate
                    best = current
                    best_count = len(current)
                else:
                    # Try swapping two comparators
                    if len(current) >= 2:
                        i, j = rng.sample(range(len(current)), 2)
                        candidate2 = list(current)
                        candidate2[i], candidate2[j] = candidate2[j], candidate2[i]
                        if _verify_network_fast(candidate2):
                            current = candidate2
                            best = current
                            best_count = len(current)
    
    # Strategy 4: Final polish - try all single removals one more time
    if time.time() < deadline - 5.0 and best_count > 55:
        current = list(best)
        for idx in range(len(current)):
            if time.time() >= deadline - 5.0:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                best = current
                best_count = len(current)
    
    # Final verification
    if not _verify_network_fast(best):
        return base
    
    return best

# EVOLVE-BLOCK-END
