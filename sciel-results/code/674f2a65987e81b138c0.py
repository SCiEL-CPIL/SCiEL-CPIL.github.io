"""Size-efficient sorting network on 16 wires."""
N = 16

# EVOLVE-BLOCK-START
def _verify(net, N):
    """Verify a sorting network using zero-one principle."""
    for mask in range(1 << N):
        inputs = [(mask >> i) & 1 for i in range(N)]
        for (i, j) in net:
            if inputs[i] > inputs[j]:
                inputs[i], inputs[j] = inputs[j], inputs[i]
        for k in range(N - 1):
            if inputs[k] > inputs[k + 1]:
                return False
    return True

def _bitonic_merge(n, offset=0):
    """Generate bitonic merge comparators for n wires starting at offset."""
    if n <= 1:
        return []
    h = n // 2
    comparators = []
    # Compare (i, i+h) for i in range(h)
    for i in range(h):
        comparators.append((offset + i, offset + i + h))
    # Recursively merge halves
    for (a, b) in _bitonic_merge(h, offset):
        comparators.append((a, b))
    for (a, b) in _bitonic_merge(h, offset + h):
        comparators.append((a, b))
    return comparators

def _bitonic_sort(n, offset=0):
    """Generate bitonic sort network for n wires starting at offset."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    h = n // 2
    comparators = []
    # Sort first half ascending
    for (a, b) in _bitonic_sort(h, offset):
        comparators.append((a, b))
    # Sort second half descending (by reversing comparator direction)
    # To sort descending, we swap the roles: compare (j, i) instead of (i, j)
    # But our comparators always put min at first wire, max at second.
    # To get descending order in the second half, we can generate the ascending
    # sort and then reverse each comparator (swap i and j).
    for (a, b) in _bitonic_sort(h, offset + h):
        comparators.append((b, a))  # This makes it sort descending
    # Bitonic merge
    for (a, b) in _bitonic_merge(n, offset):
        comparators.append((a, b))
    return comparators

def _batcher_oom(n, offset=0):
    """Generate Batcher's odd-even-merge sort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    h = n // 2
    if n % 2 != 0:
        raise ValueError("Only even n supported")
    left = _batcher_oom(h, offset)
    right = _batcher_oom(h, offset + h)
    comparators = left + right
    # Odd-even merge
    comparators += _odd_even_merge(n, offset)
    return comparators

def _odd_even_merge(n, offset=0):
    """Generate odd-even merge comparators for n wires starting at offset."""
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    h = n // 2
    comparators = []
    # Odd merge: compare (offset+1, offset+1+h), (offset+3, offset+3+h), ...
    for i in range(1, h, 2):
        comparators.append((offset + i, offset + i + h))
    # Even merge: compare (offset+0, offset+0+h), (offset+2, offset+2+h), ...
    for i in range(0, h, 2):
        comparators.append((offset + i, offset + i + h))
    # Recursively merge odd and even subsequences
    for (a, b) in _odd_even_merge(h, offset + 1):
        comparators.append((a, b))
    for (a, b) in _odd_even_merge(h, offset):
        comparators.append((a, b))
    return comparators

def construct():
    # Generate candidate networks
    candidates = []
    
    # Bitonic sort
    try:
        bitonic_net = _bitonic_sort(N)
        candidates.append(bitonic_net)
    except Exception:
        pass
    
    # Batcher odd-even-merge
    try:
        batcher_net = _batcher_oom(N)
        candidates.append(batcher_net)
    except Exception:
        pass
    
    # Find the best verified network
    best_net = None
    best_count = float('inf')
    
    for net in candidates:
        if len(net) >= best_count:
            continue
        if _verify(net, N):
            best_count = len(net)
            best_net = net
    
    if best_net is None:
        # Fallback: odd-even transposition
        best_net = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
    
    return best_net
# EVOLVE-BLOCK-END
