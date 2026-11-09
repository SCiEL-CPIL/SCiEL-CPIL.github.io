"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # Batcher odd-even merge sort for N=16
    comps = _batcher_odd_merge(16)
    # Verify with zero-one principle over all 2^16 Boolean inputs
    assert _verify(comps, 16), "Network failed zero-one verification"
    return comps

def _batcher_odd_merge(n):
    """Return comparator list for batcher's odd-even merge sort on n wires."""
    if n == 1:
        return []
    if n % 2 == 1:
        # Sort odd-positioned wires and even-positioned wires, then merge
        # For odd n, pad with a dummy
        comps = []
        # Sort first half (positions 0,2,4,...) and second half (1,3,5,...)
        # Actually for odd n, we need to handle carefully
        # Use recursive: sort odd-indexed and even-indexed subsequences
        half = n // 2
        # Even indices: 0, 2, 4, ..., n-2 (half elements)
        # Odd indices: 1, 3, 5, ..., n-1 (half elements if n even, else half)
        # For simplicity, handle even n primarily; for odd, pad
        return _batcher_odd_merge_even(n)
    return _batcher_odd_merge_even(n)

def _batcher_odd_merge_even(n):
    """Batcher odd-even merge for even n."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    # Recursively sort even positions (0, 2, 4, ...)
    even_comps = []
    for i in range(half - 1):
        for (a, b) in _batcher_odd_merge_even(half):
            even_comps.append((2 * a, 2 * b))
    # Recursively sort odd positions (1, 3, 5, ...)
    odd_comps = []
    for (a, b) in _batcher_odd_merge_even(half):
        odd_comps.append((2 * a + 1, 2 * b + 1))
    # Merge the two sorted halves using batcher's merge
    merge_comps = _batcher_merge(n)
    return even_comps + odd_comps + merge_comps

def _batcher_merge(n):
    """Batcher's merge of two sorted halves of size n/2 into sorted whole."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    comps = []
    # Compare (0, 1), (2, 3), ..., (n-2, n-1)
    for i in range(0, n - 1, 2):
        comps.append((i, i + 1))
    # Recursively merge the two resulting halves
    # Left half: elements at positions 0, 2, 4, ... (the min of each pair)
    # Right half: elements at positions 1, 3, 5, ... (the max of each pair)
    left_comps = []
    for (a, b) in _batcher_merge(half):
        left_comps.append((2 * a, 2 * b))
    right_comps = []
    for (a, b) in _batcher_merge(half):
        right_comps.append((2 * a + 1, 2 * b + 1))
    # Compare (1, 2), (3, 4), ..., (n-3, n-2)
    for i in range(1, n - 1, 2):
        comps.append((i, i + 1))
    return comps + left_comps + right_comps

def _verify(comps, n):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle."""
    for mask in range(1 << n):
        wires = [1 if (mask >> i) & 1 else 0 for i in range(n)]
        for (i, j) in comps:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check sorted
        for i in range(n - 1):
            if wires[i] > wires[i + 1]:
                return False
    return True
# EVOLVE-BLOCK-END
