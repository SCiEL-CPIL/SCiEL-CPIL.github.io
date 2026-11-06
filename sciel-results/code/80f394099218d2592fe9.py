"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time

def _verify_network(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle."""
    import itertools
    for bits in itertools.product([0, 1], repeat=n):
        vals = list(bits)
        for i, j in network:
            if vals[i] > vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
        if vals != sorted(vals):
            return False
    return True

def _batcher_odd_even_mergesort(n):
    """Generate Batcher's odd-even mergesort network for n wires.
    Returns list of (i, j) comparator pairs."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 4:
        return [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
    
    half = n // 2
    # Sort left half
    left = _batcher_odd_even_mergesort(half)
    # Sort right half (offset by half)
    right = [(i + half, j + half) for i, j in _batcher_odd_even_mergesort(half)]
    
    # Merge the two sorted halves
    # Batcher's odd-even merge:
    # 1. Compare corresponding elements: (i, i+half) for i in range(half)
    merge = [(i, i + half) for i in range(half)]
    
    # 2. After this, even-indexed positions contain the half smallest elements (sorted)
    #    and odd-indexed positions contain the half largest elements (sorted)
    # 3. Recursively sort the even-indexed subsequence and odd-indexed subsequence
    #    The even-indexed positions are 0, 2, 4, ..., 2*(half-1)
    #    Map position k in subsequence to wire 2*k
    even_subseq = _batcher_odd_even_mergesort(half)
    even_mapped = [(2 * a, 2 * b) for a, b in even_subseq]
    
    odd_subseq = _batcher_odd_even_mergesort(half)
    odd_mapped = [(2 * a + 1, 2 * b + 1) for a, b in odd_subseq]
    
    return left + right + merge + even_mapped + odd_mapped

def construct():
    """Return a sorting network for 16 wires."""
    net = _batcher_odd_even_mergesort(16)
    # Verify correctness
    if _verify_network(net, 16):
        return net
    # Fallback to odd-even transposition if verification fails
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
