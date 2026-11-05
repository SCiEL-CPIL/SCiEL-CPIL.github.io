"""Size-efficient sorting network search for 16 wires."""
import numpy as np
import time
import random
import itertools

N = 16
MAX_COMPARATORS = 4096

# EVOLVE-BLOCK-START

def _build_bitparallel_inputs():
    """Pre-compute all 2^16 Boolean inputs as a (2^N, N) uint8 array."""
    all_inputs = np.array(list(itertools.product([0, 1], repeat=N)), dtype=np.uint8)
    # Shape: (65536, 16)
    return all_inputs


def _verify_bitparallel(network, all_inputs):
    """
    Verify a sorting network using bit-parallel simulation.
    network: list of (i, j) pairs
    all_inputs: (2^N, N) uint8 array
    Returns True if network sorts all inputs.
    """
    x = all_inputs.copy()
    for i, j in network:
        a = x[:, i]
        b = x[:, j]
        # min goes to i, max goes to j
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    # Check sorted: for each column, values must be non-decreasing
    # i.e., x[k, t] <= x[k, t+1] for all k, t
    diffs = np.diff(x, axis=1)
    return np.all(diffs >= 0)


def _odd_even_transposition():
    """Classic odd-even transposition sort: N phases, N(N-1)/2 comparators."""
    comps = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _batcher_odd_even_merge(n):
    """
    Batcher's odd-even merge for n wires (assumes n is power of 2).
    Returns list of (i, j) comparators.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    result = []
    
    # First: compare (i, i + half) for i in 0..half-1
    for i in range(half):
        result.append((i, i + half))
    
    # Then: recursively merge odd-indexed sub-networks and even-indexed
    # Odd sub-network: wires at positions 1, 3, 5, ... -> map to 0, 1, 2, ...
    for (i, j) in _batcher_odd_even_merge(half):
        result.append((2 * i + 1, 2 * j + 1))
    
    # Even sub-network: wires at positions 0, 2, 4, ... -> map to 0, 1, 2, ...
    for (i, j) in _batcher_odd_even_merge(half):
        result.append((2 * i, 2 * j))
    
    return result


def _batcher_sort(n):
    """
    Batcher's odd-even merge sort for n wires.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    result = []
    
    # Recursively sort first half
    for (i, j) in _batcher_sort(half):
        result.append((i, j))
    
    # Recursively sort second half (offset by half)
    for (i, j) in _batcher_sort(half):
        result.append((i + half, j + half))
    
    # Merge the two sorted halves using odd-even merge
    for (i, j) in _batcher_odd_even_merge(n):
        result.append((i, j))
    
    return result


def _remove_redundant(net, all_inputs):
    """
    Greedy removal of redundant comparators.
    Try to remove each comparator; if the network still sorts, keep it removed.
    """
    net = list(net)
    changed = True
    while changed:
        changed = False
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            candidate = net[:idx] + net[idx + 1:]
            if _verify_bitparallel(candidate, all_inputs):
                net = candidate
                changed = True
    return net


def _local_search(net, all_inputs, deadline, best_len):
    """
    Local search: try to remove comparators, and also try swapping/adding.
    Returns improved network.
    """
    net = list(net)
    while time.time() < deadline:
        improved = False
        
        # Try removal
        indices = list(range(len(net)))
        random.shuffle(indices)
        for idx in indices:
            if time.time() > deadline:
                break
            candidate = net[:idx] + net[idx + 1:]
            if _verify_bitparallel(candidate, all_inputs):
                net = candidate
                improved = True
                break
        
        if not improved:
            # Try single-swap: swap two adjacent comparators
            if len(net) >= 2:
                swap_done = False
                for _ in range(50):  # limited attempts
                    if time.time() > deadline:
                        break
                    idx = random.randint(0, len(net) - 2)
                    candidate = net[:idx] + [net[idx + 1], net[idx]] + net[idx + 2:]
                    if _verify_bitparallel(candidate, all_inputs):
                        net = candidate
                        improved = True
                        swap_done = True
                        break
                if swap_done:
                    continue
            
            # Try adding a random comparator then removing
            if len(net) < MAX_COMPARATORS:
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                candidate = net + [(i, j)]
                if _verify_bitparallel(candidate, all_inputs):
                    net = candidate
                    # Now try to remove something
                    indices2 = list(range(len(net)))
                    random.shuffle(indices2)
                    for idx2 in indices2:
                        if time.time() > deadline:
                            break
                        cand2 = net[:idx2] + net[idx2 + 1:]
                        if _verify_bitparallel(cand2, all_inputs):
                            net = cand2
                            improved = True
                            break
        
        if not improved:
            break
    
    return net


def _hill_climb_from_start(start_net, all_inputs, deadline, best_len):
    """
    Hill-climbing local search from a starting network.
    """
    net = list(start_net)
    
    # First pass: aggressive removal
    net = _remove_redundant(net, all_inputs)
    
    # Second pass: local search with swaps and add-remove
    net = _local_search(net, all_inputs, deadline, best_len)
    
    return net


def _random_network(num_comparators):
    """Generate a random sorting network candidate."""
    comps = []
    for _ in range(num_comparators):
        i, j = random.sample(range(N), 2)
        if i > j:
            i, j = j, i
        comps.append((i, j))
    return comps


def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    all_inputs = _build_bitparallel_inputs()
    
    # Generate initial candidates
    candidates = []
    
    # Odd-even transposition
    oet = _odd_even_transposition()
    candidates.append(oet)
    
    # Batcher odd-even merge sort
    bsort = _batcher_sort(N)
    candidates.append(bsort)
    
    # Verify and filter
    valid_candidates = []
    for cand in candidates:
        if _verify_bitparallel(cand, all_inputs):
            valid_candidates.append(cand)
    
    # Sort by length
    valid_candidates.sort(key=len)
    
    best = valid_candidates[0] if valid_candidates else oet
    best_len = len(best)
    
    # Hill-climb from each valid candidate
    for start in valid_candidates:
        if time.time() > deadline - 5:
            break
        if len(start) >= best_len:
            continue
        
        net = _hill_climb_from_start(start, all_inputs, deadline - 5, best_len)
        if len(net) < best_len:
            best = net
            best_len = len(net)
    
    # Try random restarts with local search
    while time.time() < deadline - 5:
        # Start from a random network and try to make it valid via local search
        # Better: start from best and perturb
        if len(best) == 0:
            break
        
        # Perturbation: remove a random set of comparators, then try to repair
        net = list(best)
        if len(net) > 10:
            # Remove a few random comparators
            num_remove = random.randint(1, min(3, len(net) - 5))
            remove_indices = random.sample(range(len(net)), num_remove)
            for idx in sorted(remove_indices, reverse=True):
                net.pop(idx)
        
        # Try to repair by adding comparators
        if not _verify_bitparallel(net, all_inputs):
            # Add random comparators until valid (with limit)
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                net.append((i, j))
                if _verify_bitparallel(net, all_inputs):
                    break
        
        if _verify_bitparallel(net, all_inputs):
            # Local search to minimize
            net = _hill_climb_from_start(net, all_inputs, deadline - 5, best_len)
            if len(net) < best_len:
                best = net
                best_len = len(net)
    
    return best

# EVOLVE-BLOCK-END
