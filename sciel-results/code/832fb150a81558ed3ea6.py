"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 18

# EVOLVE-BLOCK-START

def _bitonic_sort_network(n):
    """Generate a bitonic sorting network for n wires.
    Returns list of (i, j) pairs where 0 <= i < j < n."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # For power-of-2 sizes, use standard recursive bitonic
    # For non-power-of-2, we still use a recursive approach
    # Split into two halves
    mid = n // 2
    left_size = mid
    right_size = n - mid
    
    # Recursively sort each half
    left = _bitonic_sort_network(left_size)
    right = _bitonic_sort_network(right_size)
    
    # Shift right network indices
    right_shifted = [(i + left_size, j + left_size) for i, j in right]
    
    # Merge the two sorted halves into a bitonic sequence
    merge = _bitonic_merge_network(n, left_size, right_size)
    
    return left + right_shifted + merge

def _bitonic_merge_network(n, left_size, right_size):
    """Merge two sorted sequences of sizes left_size and right_size into a sorted sequence.
    The two sequences are on wires [0, left_size) and [left_size, n).
    Returns comparators to merge them."""
    if n <= 1:
        return []
    if left_size == 0 or right_size == 0:
        return []
    if left_size == 1 and right_size == 1:
        return [(0, 1)]
    
    # Standard bitonic merge: compare elements i and i + min(left_size, right_size)
    # for i in range(min(left_size, right_size))
    # But we need to handle the asymmetric case properly.
    
    # For the general case, use the standard bitonic merge pattern:
    # Phase 1: compare (i, i + half) where half = n // 2
    half = n // 2
    comparators = []
    
    # Compare pairs (i, i + half) for i in range(half)
    for i in range(half):
        if i + half < n:
            comparators.append((i, i + half))
    
    # Now recursively merge the top half and bottom half
    # Top half: wires [0, half) - should be sorted
    # Bottom half: wires [half, n) - should be sorted
    # But wait, after the first round of comparisons, we have:
    # Wires [0, half) contain the smaller elements, [half, n) contain the larger
    # So we need to sort each half recursively
    
    top = _bitonic_sort_network(half)
    bottom = _bitonic_sort_network(n - half)
    bottom_shifted = [(i + half, j + half) for i, j in bottom]
    
    return comparators + top + bottom_shifted

def construct():
    """Construct a sorting network for N wires using bitonic sort.
    Returns an ordered list of (i, j) pairs."""
    start_time = time.time()
    
    # Generate the bitonic sorting network
    network = _bitonic_sort_network(N)
    
    # Verify all pairs are valid
    for i, j in network:
        assert 0 <= i < j < N, f"Invalid comparator ({i}, {j})"
    
    return network

# EVOLVE-BLOCK-END
