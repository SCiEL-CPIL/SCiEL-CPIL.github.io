"""Size-efficient sorting networks on 18 wires.
Uses a hybrid approach: start with a known good network (odd-even transposition),
then apply local search / simulated annealing to reduce comparator count while
maintaining correctness over all 2^18 Boolean inputs.
"""

import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N  # 262144

# Deadline guard
DEADLINE = 85.0  # seconds (5s safety margin before 90s kill)


def _precompute_sorted_mask():
    """Precompute a boolean array: True if the state index is a sorted configuration."""
    mask = np.zeros(NUM_BOOLS, dtype=np.bool_)
    for k in range(N + 1):
        # k ones in the top k positions (wires N-k through N-1)
        val = ((1 << k) - 1) << (N - k)
        mask[val] = True
    return mask


# Precompute sorted mask at module level
_SORTED_MASK = _precompute_sorted_mask()


def _verify_network(comparators):
    """Verify that the network sorts all 2^18 Boolean inputs.
    Returns True if valid, False otherwise.
    """
    current = np.arange(NUM_BOOLS, dtype=np.int32)
    for i, j in comparators:
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        # Where bit_i=1 and bit_j=0, swap bits i and j
        mask_swap = (bit_i == 1) & (bit_j == 0)
        # Swap: clear both bits, set j to 1, set i to 0... wait.
        # If bit_i=1, bit_j=0: after swap, bit_i=0, bit_j=1
        # So: clear bit_i, set bit_j
        # current = current & ~(1<<i) | (1<<j) for those where swap happens
        # But only where mask_swap is True
        clear_i = ~(1 << i)
        set_j = (1 << j)
        current = np.where(mask_swap, (current & clear_i) | set_j, current)
    
    # Check if all are sorted
    return np.all(_SORTED_MASK[current])


def _odd_even_transposition_network():
    """Generate the odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _random_comparator():
    """Generate a random comparator (i, j) with i < j."""
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)


def _local_search(initial_comparators, max_time=80.0, rng_seed=0):
    """Local search to reduce comparator count while maintaining validity."""
    start_time = time.time()
    rng = random.Random(rng_seed)
    best_comparators = list(initial_comparators)
    best_len = len(best_comparators)
    
    current = list(best_comparators)
    
    # Phase 1: Try removing each comparator (from end to start)
    for idx in range(len(current) - 1, -1, -1):
        if time.time() - start_time > max_time:
            break
        if len(current) <= 68:  # Can't go below lower bound
            break
        
        trial = current[:idx] + current[idx+1:]
        if _verify_network(trial):
            current = trial
            if len(current) < best_len:
                best_len = len(current)
                best_comparators = list(current)
    
    # Phase 2: Random removal attempts
    improved = True
    while improved:
        if time.time() - start_time > max_time:
            break
        improved = False
        if len(current) <= 68:
            break
        
        # Try removing a random comparator
        idx = rng.randint(0, len(current) - 1)
        trial = current[:idx] + current[idx+1:]
        if _verify_network(trial):
            current = trial
            improved = True
            if len(current) < best_len:
                best_len = len(current)
                best_comparators = list(current)
        
        # Try swapping adjacent comparators
        if len(current) >= 2:
            idx = rng.randint(0, len(current) - 2)
            trial = current[:]
            trial[idx], trial[idx+1] = trial[idx+1], trial[idx]
            if _verify_network(trial):
                current = trial
                improved = True
    
    # Phase 3: Try replacing pairs with single comparators
    for _ in range(100):  # Limited iterations
        if time.time() - start_time > max_time:
            break
        if len(current) <= 68:
            break
        
        idx = rng.randint(0, max(0, len(current) - 2))
        if idx >= len(current) - 1:
            continue
        
        new_comp = _random_comparator()
        trial = current[:idx] + [new_comp] + current[idx+2:]
        if _verify_network(trial):
            current = trial
            if len(current) < best_len:
                best_len = len(current)
                best_comparators = list(current)
    
    return best_comparators


def construct():
    """Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with odd-even transposition network
    comparators = _odd_even_transposition_network()
    
    # Verify baseline
    if not _verify_network(comparators):
        # Should not happen, but fallback
        return comparators
    
    # Try local search to reduce size with multiple restarts
    best_net = list(comparators)
    best_len = len(best_net)
    
    # Multiple restarts with different seeds
    for seed in range(5):
        if time.time() > deadline - 10:
            break
        
        remaining = max(5, deadline - time.time() - 10)
        net = _local_search(comparators, max_time=remaining, rng_seed=seed)
        
        if len(net) < best_len:
            best_len = len(net)
            best_net = list(net)
    
    # Final verification
    if not _verify_network(best_net):
        # Fallback to known good network
        best_net = _odd_even_transposition_network()
    
    return best_net
