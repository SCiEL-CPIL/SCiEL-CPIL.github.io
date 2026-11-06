N = 16

# EVOLVE-BLOCK-START
import time

def _sort2():
    return [(0, 1)]

def _sort3():
    return [(0, 1), (1, 2), (0, 1)]

def _sort4():
    """Optimal 4-input sorting network with 5 comparators."""
    return [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)]

def _sort8():
    """Optimal 8-input sorting network with 19 comparators."""
    return [
        (0, 4), (1, 5), (2, 6), (3, 7),
        (0, 2), (1, 3), (4, 6), (5, 7),
        (0, 1), (2, 3), (4, 5), (6, 7),
        (1, 4), (2, 5), (3, 6),
        (1, 2), (4, 5),
        (2, 4)
    ]

def _oe_merge(m, k):
    """
    Odd-even merge two sorted sequences of length m and k.
    Assumes the m elements are at positions 0..m-1 and k elements at m..m+k-1.
    Returns list of comparators (i, j) with 0 <= i < j < m+k.
    """
    if m == 0 or k == 0:
        return []
    if m == 1 and k == 1:
        return [(0, 1)]
    
    net = []
    # Compare (i, i+m) for i in range(min(m, k))
    for i in range(min(m, k)):
        net.append((i, i + m))
    
    # Now recursively merge even and odd subsequences
    # Even positions: 0, 2, 4, ... 
    # Odd positions: 1, 3, 5, ...
    
    if m == k:
        # Even subsequence: merge of two sorted sequences of length m//2
        even_net = _oe_merge(m // 2, m // 2)
        for i, j in even_net:
            net.append((2 * i, 2 * j))
        
        # Odd subsequence: merge of two sorted sequences of length m//2
        odd_net = _oe_merge(m // 2, m // 2)
        for i, j in odd_net:
            net.append((2 * i + 1, 2 * j + 1))
    else:
        # Unequal case: handle by padding or using a different strategy
        # For simplicity, if sizes differ, use a fallback
        # This shouldn't happen in our 16-wire case since we split evenly
        # But let's handle it: the even subsequence has ceil((m+k)/2) elements
        # and odd has floor((m+k)/2)
        # For now, if m != k, just do a full sort as fallback
        n = m + k
        if n <= 4:
            # Use known small networks
            if n == 2:
                net = [(0, 1)]
            elif n == 3:
                net = [(0, 1), (1, 2), (0, 1)]
            elif n == 4:
                net = _sort4()
        else:
            # Fallback: odd-even transposition
            net = [(i, i + 1) for phase in range(n)
                   for i in range(phase % 2, n - 1, 2)]
    
    return net

def _verify_network(network, n=16):
    """Verify the network sorts all 2^n inputs using zero-one principle."""
    for mask in range(1 << n):
        inputs = [(mask >> i) & 1 for i in range(n)]
        for (i, j) in network:
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
        for i in range(n - 1):
            if inputs[i] > inputs[i + 1]:
                return False
    return True

def _known_16_network():
    """
    Construct a 16-input sorting network using Batcher's odd-even merge sort.
    Step 1: Sort two halves of 8 wires each.
    Step 2: Compare (i, i+8) for i in 0..7.
    Step 3: Odd-even merge the even and odd subsequences.
    """
    comparators = []
    
    # Step 1: Sort the two halves
    for (i, j) in _sort8():
        comparators.append((i, j))
    for (i, j) in _sort8():
        comparators.append((i + 8, j + 8))
    
    # Step 2: Compare (i, i+8) for i in range(8)
    for i in range(8):
        comparators.append((i, i + 8))
    
    # Step 3: Odd-even merge even and odd subsequences
    # Even positions: 0, 2, 4, 6, 8, 10, 12, 14 (8 elements)
    # These form a merge of two sorted sequences of 4 elements each
    even_net = _oe_merge(4, 4)
    for i, j in even_net:
        comparators.append((2 * i, 2 * j))
    
    # Odd positions: 1, 3, 5, 7, 9, 11, 13, 15 (8 elements)
    odd_net = _oe_merge(4, 4)
    for i, j in odd_net:
        comparators.append((2 * i + 1, 2 * j + 1))
    
    return comparators

def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Build the Batcher odd-even merge sort network
    network = _known_16_network()
    
    # Verify it
    if _verify_network(network):
        return network
    
    # Fallback: odd-even transposition sort (always correct)
    network = [(i, i + 1) for phase in range(N)
               for i in range(phase % 2, N - 1, 2)]
    
    return network
# EVOLVE-BLOCK-END
