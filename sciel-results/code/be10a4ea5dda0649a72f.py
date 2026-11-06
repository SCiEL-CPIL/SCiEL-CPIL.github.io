"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 18

# EVOLVE-BLOCK-START
import time
import sys

def _bitonic_sort_16():
    """Generates Batcher's Bitonic Sort network for 16 wires."""
    comparators = []
    
    def _bitonic_merge(n, offset, direction):
        # direction: 1 for ascending, -1 for descending
        if n <= 1:
            return
        k = n // 2
        # Compare pairs
        for i in range(k):
            a = offset + i
            b = offset + i + k
            if direction == 1:
                # min to a, max to b
                comparators.append((a, b))
            else:
                # max to a, min to b
                comparators.append((a, b)) # Comparator always min to lower index
                # To get descending, we need to swap the logic?
                # No, standard comparator is min->i, max->j.
                # For descending merge, we want max->i, min->j.
                # So we compare and swap if min > max?
                # In a sorting network, the comparator is fixed: min to lower index.
                # So to sort descending, we must arrange the wires such that
                # the "lower" index in the comparator holds the larger value?
                # No, the wire indices are fixed.
                # To sort descending, we use comparators that put max in lower index?
                # Standard comparators put min in lower index.
                # So we cannot directly create a descending sort with standard comparators
                # unless we reverse the wire order?
                # Batcher's Bitonic Sort uses standard comparators.
                # The "direction" is handled by the structure of the network.
                # Actually, for bitonic merge, if we want descending,
                # we compare (i, i+k) and the min goes to i, max to i+k.
                # This results in the first half being smaller?
                # No, if we want descending, the first half should be larger.
                # So we need max in i, min in i+k.
                # This requires a comparator that puts max in lower index.
                # But the interface says: "Each compare-exchange writes min to wire i and max to j".
                # So we CANNOT put max in lower index.
                # Therefore, we can ONLY sort in ascending order using standard comparators.
                # Bitonic Sort for ascending only?
                # Yes, standard Bitonic Sort produces ascending order.
                pass
        
        # Recursively merge halves
        _bitonic_merge(k, offset, direction)
        _bitonic_merge(k, offset + k, direction)

    # This implementation is tricky with the direction constraint.
    # Standard Bitonic Sort for ascending:
    # 1. Split into halves, sort ascending.
    # 2. Merge.
    
    # Let's use a known correct generator for Bitonic Sort (Ascending).
    def _gen_bitonic(n, offset):
        if n <= 1:
            return
        k = n // 2
        _gen_bitonic(k, offset)
        _gen_bitonic(k, offset + k)
        _merge_bitonic(n, offset, 1) # 1 for ascending

    def _merge_bitonic(n, offset, direction):
        if n <= 1:
            return
        k = n // 2
        for i in range(k):
            comparators.append((offset + i, offset + i + k))
        _merge_bitonic(k, offset, direction)
        _merge_bitonic(k, offset + k, direction)

    _gen_bitonic(16, 0)
    return comparators

def _insert_one_into_sorted(n, x_wire, arr_offset):
    """
    Inserts x_wire into sorted array at arr_offset..arr_offset+n-1.
    Returns comparators.
    """
    if n == 0:
        return []
    if n == 1:
        # Compare x_wire and arr_offset
        # Min goes to lower index.
        if x_wire < arr_offset:
            return [(x_wire, arr_offset)]
        else:
            return [(arr_offset, x_wire)]
    
    mid = n // 2
    mid_wire = arr_offset + mid
    
    # Compare x_wire and mid_wire
    # Min goes to min(x_wire, mid_wire), Max to max(x_wire, mid_wire)
    # We don't know which is which.
    # This simple recursive insertion does NOT work with fixed comparators
    # because the "left" and "right" values move to different wires.
    
    # The correct "Insert 1 into N" network is a tree.
    # It has size N.
    # Structure:
    # Compare x with A[n/2].
    # The min stays in the "array" part?
    # No, the min goes to the lower index.
    # If x < A[n/2], x goes to A[n/2]'s wire? No, x is on x_wire.
    # If x_wire < A[n/2], min is x, goes to x_wire.
    # If x_wire > A[n/2], min is A[n/2], goes to A[n/2].
    
    # This is complex. Let's use a simpler, valid network.
    # Odd-Even Transposition is 153.
    # Let's just return the Bitonic 16 + Sort 2 + Transposition Merge?
    # Merge 16 and 2 using Transposition?
    # Just run Transposition on all 18? That's 153.
    
    # Let's try to get a valid improvement.
    # Use Bitonic 16 (80 comps).
    # Sort 2 (1 comp).
    # Merge 16 and 2 using a simple network.
    # We can use the fact that 16 is sorted and 2 is sorted.
    # We can use a "Merge" network for 16 and 2.
    # Since 2 is small, we can just compare each of the 2 elements with the 16?
    # No, that's 32 comps.
    # Total 80 + 1 + 32 = 113.
    # Is comparing each element of B with all of A sufficient?
    # No, that's not a sorting network.
    
    # Let's use the **Odd-Even Merge** for 18.
    # I will implement a basic Odd-Even Merge.
    pass

def construct():
    # Use a known efficient network.
    # For now, return the Bitonic 16 + Sort 2 + Simple Merge (32 comps)
    # This is 113 comparators.
    
    comps = _bitonic_sort_16()
    comps.append((16, 17))
    
    # Merge 16 and 2.
    # We have sorted A[0..15] and B[16, 17].
    # We can use a "Merge 1 and N" for each element of B.
    # But we need a fixed network.
    
    # Let's use a standard "Merge" network for 16 and 2.
    # It is not trivial.
    
    # Fallback: Use Odd-Even Transposition but prune?
    # No.
    
    # Let's return the 16-bitonic + 2-sort + 32 comparators (compare each B with all A).
    # This is NOT correct.
    
    # I will return the seed program if I can't find a valid improvement quickly.
    # But 153 is the seed.
    
    # Let's try **Batcher's Bitonic Sort** for 18 by using 32 wires and ignoring 14?
    # No.
    
    # I will use **Odd-Even Transposition** but with a **parallel** depth reduction?
    # No, size is the score.
    
    # Let's use **Shellsort**.
    # Shellsort network for 18.
    
    # Given the difficulty, I will stick to the seed to avoid invalidity.
    # However, the seed is 153.
    
    # Wait, I can use **Batcher's Bitonic Sort** for 16 (80) and then
    # **Insert** the 18th wire using a **Galperin-Miller** network?
    # I don't have the code for that.
    
    # I will return the seed.
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
# EVOLVE-BLOCK-END
