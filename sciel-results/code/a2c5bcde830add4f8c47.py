"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    """Return a fixed compare-exchange network that sorts 16 wires.
    
    Uses Batcher's Bitonic Sort network for N=16.
    The bitonic sort network for 2^k elements uses:
    C(2^k) = C(2^(k-1)) * 2 + (k-1)*2^(k-1)
    For k=4 (16 wires):
    C(2) = 1
    C(4) = 1*2 + 1*2 = 4
    C(8) = 4*2 + 2*4 = 16
    C(16) = 16*2 + 3*8 = 68
    
    Wait, let me recalculate. The standard bitonic sort comparator count for n=2^k is:
    B(2) = 1
    B(4) = 4
    B(8) = 16
    B(16) = 52
    
    Let me implement it directly.
    """
    comparators = []
    
    def bitonic_merge(n, start, direction):
        """Merge two bitonic sequences of length n/2 each.
        If direction is 1, sort in ascending order; if 0, descending.
        """
        if n <= 1:
            return
        step = n // 2
        for i in range(start, start + step):
            j = i + step
            if direction == 1:
                # Sort ascending: min to i, max to j
                comparators.append((i, j))
            else:
                # Sort descending: max to i, min to j
                # But we can only do min to i, max to j
                # So for descending, we swap the indices
                comparators.append((j, i))
        # Now recursively sort each half
        bitonic_merge(step, start, 1)
        bitonic_merge(step, start + step, 0)
    
    def bitonic_sort(n, start):
        """Sort n elements starting at index start."""
        if n <= 1:
            return
        half = n // 2
        # First, create bitonic sequences by sorting first half ascending, second half descending
        bitonic_sort(half, start)
        # For the second half, we need to sort in reverse
        # We'll handle this by building up bitonic sequences
        
        # Actually, let me use a cleaner recursive approach
    
    # Let me implement it more carefully
    comparators = []
    
    def bitonic_sort_rec(n, offset, reverse=False):
        """Sort n elements starting at offset.
        If reverse is False, sort ascending.
        If reverse is True, sort descending.
        """
        if n <= 1:
            return
        
        half = n // 2
        
        # Recursively sort the two halves, one in each direction to create bitonic sequence
        bitonic_sort_rec(half, offset, not reverse)
        bitonic_sort_rec(half, offset + half, reverse)
        
        # Now merge the two sorted (one asc, one desc) halves into a fully sorted sequence
        for i in range(offset, offset + half):
            j = i + half
            if not reverse:
                comparators.append((i, j))
            else:
                comparators.append((j, i))
    
    # Sort all 16 elements in ascending order
    bitonic_sort_rec(16, 0, False)
    
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for comp in comparators:
        if comp not in seen:
            seen.add(comp)
            unique.append(comp)
    
    return unique
# EVOLVE-BLOCK-END
