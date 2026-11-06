"""
Sorting network optimization for N=18 using:
1. Fast bitmask-based verification
2. Batcher's odd-even mergesort as starting point
3. Greedy removal with randomized restarts
4. Local search with comparator swaps
"""
import time
import random
import numpy as np

N = 18
MAX_COMPARATORS = 4096
DEADLINE_SECONDS = 85  # Safety margin before 90s kill


def _depth(net):
    """Compute the parallel depth of a comparator network."""
    if not net:
        return 0
    layer = [0] * N
    for i, j in net:
        layer[i] = max(layer[i], layer[j]) + 1
    return max(layer) if layer else 0


def _verify_network(net):
    """
    Verify that the comparator network sorts all 2^N boolean inputs.
    Uses the 0-1 principle with vectorized NumPy operations.
    """
    n = len(net)
    if n > MAX_COMPARATORS:
        return False
    if n == 0:
        return True
    
    # Generate all 2^N boolean inputs as a matrix
    all_masks = np.arange(1 << N, dtype=np.int32)
    wires = np.zeros((1 << N, N), dtype=np.int8)
    for i in range(N):
        wires[:, i] = (all_masks >> i) & 1
    
    # Apply comparators using vectorized min/max
    for i, j in net:
        wi = wires[:, i]
        wj = wires[:, j]
        min_val = np.minimum(wi, wj)
        max_val = np.maximum(wi, wj)
        wires[:, i] = min_val
        wires[:, j] = max_val
    
    # Check if sorted: wires[:, k] <= wires[:, k+1] for all k
    if np.any(wires[:, :-1] > wires[:, 1:]):
        return False
    return True


def _odd_even_transposition():
    """Generate the odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _batcher_odd_merge(n):
    """
    Generate Batcher's odd-even merge network for n inputs.
    Returns list of comparators.
    """
    if n <= 1:
        return []
    
    # Split into even and odd indexed inputs
    even_idx = list(range(0, n, 2))
    odd_idx = list(range(1, n, 2))
    
    net = []
    
    # Recursively sort even and odd subsequences
    if len(even_idx) > 1:
        sub_even = _batcher_odd_merge(len(even_idx))
        for i, j in sub_even:
            net.append((even_idx[i], even_idx[j]))
    
    if len(odd_idx) > 1:
        sub_odd = _batcher_odd_merge(len(odd_idx))
        for i, j in sub_odd:
            net.append((odd_idx[i], odd_idx[j]))
    
    # Merge the two sorted subsequences
    m = n // 2
    # Compare elements from even and odd subsequences
    for k in range(m):
        if k < len(odd_idx):
            net.append((even_idx[k], odd_idx[k]))
    
    # Recursive merges on adjacent pairs
    if m > 1:
        # Merge (even[0], odd[0]) with (even[1], odd[1]), etc.
        for k in range(1, m):
            if k < len(even_idx) and (k-1) < len(odd_idx):
                net.append((even_idx[k-1], odd_idx[k-1]))
            if k < len(odd_idx) and (k-1) < len(even_idx):
                net.append((odd_idx[k-1], even_idx[k-1]))
    
    return net


def _batcher_mergesort(n):
    """
    Generate Batcher's odd-even mergesort network for n inputs.
    """
    if n <= 1:
        return []
    
    if n == 2:
        return [(0, 1)]
    
    # Split into two halves
    half = n // 2
    net = []
    
    # Recursively sort each half
    left_net = _batcher_mergesort(half)
    for i, j in left_net:
        net.append((i, j))
    
    right_net = _batcher_mergesort(n - half)
    for i, j in right_net:
        net.append((i + half, j + half))
    
    # Merge the two sorted halves using odd-even merge
    merge_net = _batcher_odd_merge(n)
    net.extend(merge_net)
    
    return net


def _greedy_remove(net, deadline):
    """
    Greedy removal: repeatedly try to remove comparators until no more can be removed.
    """
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        
        # Try removing from end to start
        for idx in range(len(net) - 1, -1, -1):
            if time.time() > deadline:
                return net
            
            new_net = net[:idx] + net[idx+1:]
            
            if _verify_network(new_net):
                net = new_net
                changed = True
                break
    
    return net


def _local_search(net, deadline, max_iters=200):
    """
    Local search: try replacing individual comparators with random ones.
    """
    for _ in range(max_iters):
        if time.time() > deadline:
            break
        if len(net) == 0:
            break
        
        # Pick a random comparator to replace
        idx = random.randint(0, len(net) - 1)
        i, j = net[idx]
        
        # Try a random different comparator
        ni = random.randint(0, N - 2)
        nj = random.randint(ni + 1, N - 1)
        if (ni, nj) == (i, j):
            continue
        
        new_net = net[:]
        new_net[idx] = (ni, nj)
        
        if _verify_network(new_net):
            net = new_net
    
    return net


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # Start with Batcher's odd-even mergesort (typically ~100-120 comparators for N=18)
    # Fallback to odd-even transposition if needed
    try:
        net = _batcher_mergesort(N)
    except Exception:
        net = _odd_even_transposition()
    
    # If Batcher's is worse, use odd-even transposition
    oet = _odd_even_transposition()
    if len(net) > len(oet):
        net = oet
    
    best_net = net[:]
    best_count = len(net)
    
    # Phase 1: Greedy removal
    net = _greedy_remove(net, deadline - 10)
    if len(net) < best_count:
        best_net = net[:]
        best_count = len(net)
    
    # Phase 2: Local search
    if time.time() < deadline - 5:
        net = _local_search(net, deadline - 3, max_iters=300)
        if len(net) < best_count:
            best_net = net[:]
            best_count = len(net)
    
    # Phase 3: Try starting from odd-even transposition if we haven't beaten it
    if best_count >= len(oet) and time.time() < deadline - 15:
        net2 = oet[:]
        net2 = _greedy_remove(net2, deadline - 5)
        if len(net2) < best_count:
            best_net = net2[:]
            best_count = len(net2)
        
        if time.time() < deadline - 3:
            net2 = _local_search(net2, deadline - 2, max_iters=100)
            if len(net2) < best_count:
                best_net = net2[:]
                best_count = len(net2)
    
    return best_net
