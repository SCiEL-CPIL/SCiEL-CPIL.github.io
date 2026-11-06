"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute input matrix
# Shape: (2^N, N)
_indices = np.arange(1 << N, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = (_indices[:, None] & _bits[None, :]) > 0
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)


def is_sorted_network(comparators):
    """
    Check if the network sorts all 2^17 inputs using the zero-one principle.
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


def _odd_even_transposition():
    """Generate the seed odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            net.append((i, i + 1))
    return net


def _bitonic_merge(start, length, ascending=True):
    """
    Generate comparators for bitonic merge of a sub-array.
    start: starting wire index
    length: length of the sub-array (must be power of 2)
    ascending: if True, merge to ascending; else to descending
    """
    comps = []
    k = length
    while k > 1:
        half = k // 2
        for i in range(start, start + length, k):
            for j in range(half):
                a = i + j
                b = i + j + half
                if ascending:
                    comps.append((a, b))
                else:
                    comps.append((a, b))  # same pair, min goes to a, max to b
                    # For descending bitonic merge, we want the larger to go to a
                    # But compare-exchange always puts min in first arg, max in second.
                    # So for descending, we swap the arguments.
                    # Actually, let's just use the same comparator; the bitonic sort
                    # algorithm handles direction by swapping inputs before merge.
                    # Let's stick to standard: compare(a,b) with a<b.
                    # For descending merge, we compare(a, b) but we want max in a.
                    # So we should compare(b, a) instead.
        k = half

    # Correct bitonic merge
    comps = []
    k = length
    while k > 1:
        half = k // 2
        for i in range(start, start + length, k):
            for j in range(half):
                a = i + j
                b = i + j + half
                if ascending:
                    comps.append((a, b))
                else:
                    comps.append((b, a))
        k = half
    return comps


def _bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires (n must be power of 2).
    """
    if n == 1:
        return []

    comps = []
    # Build bitonic sequences
    for k in range(2, n + 1, 2):
        half = k // 2
        for start in range(0, n, k):
            # First half ascending
            comps.extend(_bitonic_merge(start, half, True))
            # Second half descending
            comps.extend(_bitonic_merge(start + half, half, False))
            # Merge the two
            comps.extend(_bitonic_merge(start, k, True))
    return comps


def _hybrid_seed_network():
    """
    Build a hybrid seed network:
    - Use bitonic sort for the first 16 wires (wires 0..15)
    - Use odd-even transposition for all 17 wires to bring wire 16 into place
    - Then use bitonic sort again for all 16 (or a smarter merge)
    
    Actually, let's try a different approach:
    1. Sort wires 0..15 with bitonic (16 wires)
    2. Insert wire 16 into the sorted array using a binary-search-like network
       (or just append it and re-sort the last part)
    
    For now, let's combine:
    - Bitonic for 16 wires
    - Then a few comparators to integrate wire 16
    - Then a final bitonic-like cleanup
    
    Simpler: use odd-even transposition as base, but try to optimize.
    Let's generate a few candidate seeds and pick the best.
    """
    candidates = []
    
    # Candidate 1: Pure odd-even transposition
    candidates.append(_odd_even_transposition())
    
    # Candidate 2: Bitonic for 16 + integrate 17th wire
    # Sort 0..15 with bitonic
    bitonic_16 = _bitonic_sort_network(16)
    # Now wire 16 is unsorted. We need to merge it in.
    # A simple way: do odd-even transposition for a few phases on all 17 wires
    # to bring wire 16 into its sorted position.
    # Or, add comparators (15,16), (14,15), (13,14), ... in some order.
    
    # Let's add a "merge" network for wire 16 into the sorted 16.
    # The worst case is wire 16 is the smallest or largest.
    # We can use a binary insertion:
    # Compare (15, 16), then (14, 15) or (14, 16) depending on result...
    # But networks are non-adaptive.
    
    # Alternative: Use the fact that after bitonic sort of 16, the array is sorted.
    # We can add a "compare-swap chain" to insert wire 16:
    # (15, 16), (14, 15), (13, 14), ..., (0, 1)
    # This is 16 comparators. Total: 120 (bitonic) + 16 = 136. Same as OET.
    
    # Let's try a different bitonic variant or a known good network.
    # For now, return the odd-even transposition as it's proven correct.
    return candidates[0]


def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s kill

    # 1. Generate Seed Network
    seed_network = _odd_even_transposition()
    if not is_sorted_network(seed_network):
        return seed_network

    current = list(seed_network)
    best = list(current)
    best_len = len(current)

    # 2. Aggressive Pruning Phase (Greedy Removal)
    # Try to remove any comparator that is redundant.
    # We do this in multiple passes.
    changed = True
    while changed and time.time() < deadline - 15.0:
        changed = False
        for idx in range(len(current) - 1, -1, -1):
            if time.time() > deadline - 15.0:
                break
            candidate = current.copy()
            candidate.pop(idx)
            if is_sorted_network(candidate):
                current = candidate
                best = current.copy()
                best_len = len(current)
                changed = True
                break  # Restart the pass

    # 3. Simulated Annealing / Local Search
    T = 3.0
    T_min = 0.05
    cooling_rate = 0.99

    current = best.copy()
    current_len = best_len

    # Pre-generate a pool of random comparators to speed up "Add" and "Replace"
    all_pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    random.shuffle(all_pairs)
    pool_idx = 0

    def get_random_comparator():
        nonlocal pool_idx
        if pool_idx >= len(all_pairs):
            random.shuffle(all_pairs)
            pool_idx = 0
        comp = all_pairs[pool_idx]
        pool_idx += 1
        return comp

    steps = 0
    while time.time() < deadline:
        if T < T_min:
            break

        # Generate a neighbor
        move_type = random.randint(0, 4)
        neighbor = current.copy()

        if move_type == 0:  # Remove
            if not neighbor:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            neighbor.pop(idx)

        elif move_type == 1:  # Add
            i, j = get_random_comparator()
            idx = random.randint(0, len(neighbor))
            neighbor.insert(idx, (i, j))

        elif move_type == 2:  # Swap
            if len(neighbor) < 2:
                continue
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]

        elif move_type == 3:  # Move
            if len(neighbor) < 2:
                continue
            i1 = random.randint(0, len(neighbor) - 1)
            i2 = random.randint(0, len(neighbor) - 1)
            if i1 != i2:
                item = neighbor.pop(i1)
                neighbor.insert(i2, item)

        else:  # Replace
            if not neighbor:
                continue
            idx = random.randint(0, len(neighbor) - 1)
            i, j = get_random_comparator()
            neighbor[idx] = (i, j)

        # Check validity
        if is_sorted_network(neighbor):
            neighbor_len = len(neighbor)
            delta_e = neighbor_len - current_len

            if delta_e <= 0:
                # Improvement or equal
                current = neighbor
                current_len = neighbor_len
                if current_len < best_len:
                    best = current.copy()
                    best_len = current_len
            else:
                # Worse, accept with probability
                prob = np.exp(-delta_e / T)
                if random.random() < prob:
                    current = neighbor
                    current_len = neighbor_len
        else:
            # Invalid, reject
            pass

        T *= cooling_rate
        steps += 1

        # If no improvement in last 5000 steps, restart with a slightly perturbed best
        # (This is a simple strategy; more sophisticated would track last_improve)

    # Final check
    if is_sorted_network(best):
        return best
    else:
        return seed_network
