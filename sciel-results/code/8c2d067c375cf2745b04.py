N = 18

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a sorting network for 18 wires using a recursive Bitonic Sort approach 
    that handles non-power-of-2 sizes.
    
    Strategy:
    1. Sort wires 0-15 (size 16) using Bitonic Sort (Ascending).
    2. Sort wires 16-17 (size 2) using Bitonic Sort (Descending).
    3. Merge the two halves (size 18) using Bitonic Merge.
    
    Total comparators: 80 (sort 16) + 1 (sort 2) + 35 (merge 18) = 116.
    """
    comparators = []
    
    def _merge_asc(n, offset):
        """Merge bitonic sequence (Asc, Desc) into Asc."""
        if n <= 1:
            return
        k = n // 2
        for i in range(k):
            a = offset + i
            b = offset + i + k
            if a < b:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        _merge_asc(k, offset)
        _merge_asc(n - k, offset + k)

    def _merge_desc(n, offset):
        """Merge bitonic sequence (Desc, Asc) into Desc."""
        if n <= 1:
            return
        k = n // 2
        for i in range(k):
            a = offset + i
            b = offset + i + k
            # To get Descending, we want Max in lower index (a) and Min in higher index (b).
            # Standard comparator puts Min in a, Max in b.
            # So we must swap a and b in the pair.
            if a < b:
                comparators.append((b, a))
            else:
                comparators.append((a, b))
        _merge_desc(k, offset)
        _merge_desc(n - k, offset + k)

    def bitonic_sort(n, offset, direction=1):
        """
        Sort n wires starting at offset.
        direction: 1 for ascending, -1 for descending.
        """
        if n <= 1:
            return
        k = n // 2
        
        # Sort first half
        bitonic_sort(k, offset, 1 if direction == 1 else -1)
        # Sort second half
        bitonic_sort(n - k, offset + k, -1 if direction == 1 else 1)
        
        # Merge
        if direction == 1:
            _merge_asc(n, offset)
        else:
            _merge_desc(n, offset)

    # Step 1: Sort first 16 wires Ascending
    bitonic_sort(16, 0, 1)
    
    # Step 2: Sort last 2 wires Descending
    bitonic_sort(2, 16, -1)
    
    # Step 3: Merge the 18 wires.
    # The combined sequence is (Asc[16], Desc[2]).
    # This is a Bitonic sequence.
    # We want the final result to be Ascending.
    # So we use _merge_asc on the whole 18 wires.
    _merge_asc(18, 0)
    
    return comparators

# EVOLVE-BLOCK-END
