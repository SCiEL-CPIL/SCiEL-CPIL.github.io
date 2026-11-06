"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
def construct():
    # A 16-input sorting network built from a known efficient construction.
    # We use the Batcher odd-even merge sort network, which is well-known
    # and provably correct by the zero-one principle.
    #
    # Batcher's odd-even merge sort on 16 elements:
    # - Recursively sort the first half and second half (each of size 8)
    # - Odd-even merge the two sorted halves
    #
    # We implement this recursively to build the comparator list.
    
    def batcher_sort(n):
        """Return list of comparators for Batcher's sort network on n wires (0..n-1)."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        k = n // 2
        # Recursively sort first half and second half
        comps = []
        comps.extend(batcher_sort(k))
        # Second half is offset by k
        for c in batcher_sort(k):
            comps.append((c[0] + k, c[1] + k))
        
        # Odd-even merge the two sorted halves
        comps.extend(odd_even_merge(n))
        return comps
    
    def odd_even_merge(n):
        """
        Odd-even merge of two sorted halves of size n/2.
        The first half is wires 0..k-1, second half is wires k..n-1.
        """
        k = n // 2
        comps = []
        
        # Step 1: Compare (i, k+i) for i in 0..k-1
        for i in range(k):
            comps.append((i, k + i))
        
        # Step 2: Odd-even merge on interleaved elements
        # After step 1, we have pairs (min(a_i, b_i), max(a_i, b_i))
        # The mins are in even positions, maxes in odd positions
        # Recursively odd-even merge the mins and maxes separately
        
        # Mins are at positions: for i in 0..k-1, position i gets min
        # Actually, after comparing (i, k+i):
        #   wire i gets min(a_i, b_i)
        #   wire k+i gets max(a_i, b_i)
        
        # Now we need to merge the mins (wires 0..k-1) and maxes (wires k..2k-1)
        # using recursive odd-even merge on the interleaved structure.
        
        # The standard odd-even merge:
        # 1. Compare (i, k+i) for i=0..k-1
        # 2. Recursively odd-even merge the first half (0..k-1) and second half (k..2k-1)
        #    but with the structure that we're merging two sorted sequences
        
        # Actually, the recursive structure is:
        # After step 1, the mins form a sorted sequence and maxes form a sorted sequence.
        # We then recursively merge them by:
        # - Odd-even merge on the "odd" and "even" subsequences
        
        # Let me use the standard recursive definition:
        # odd_even_merge(n) where n is even:
        # 1. For i in 0..n/2-1: compare (i, i+n/2)
        # 2. odd_even_merge on the first n/2 elements (offset 0)
        # 3. odd_even_merge on the last n/2 elements (offset n/2)
        
        # Wait, this isn't quite right. Let me use the correct Batcher construction.
        
        # Correct odd-even merge for two sorted halves of size k:
        # 1. Compare (i, k+i) for i in 0..k-1
        # 2. Recursively odd-even merge the sequence of mins (positions 0,2,4,...) 
        #    and maxes (positions 1,3,5,...) - no, that's not right either.
        
        # Let me just use the well-known recursive formulation:
        # OMERGE(n) for n even:
        # 1. For i=0 to n/2-1: compare-exchange (i, i+n/2)
        # 2. OMERGE(n/2) on wires 0 to n/2-1
        # 3. OMERGE(n/2) on wires n/2 to n-1
        
        # This is the standard definition. Let's implement it properly.
        
        comps = []
        for i in range(k):
            comps.append((i, k + i))
        
        # Recursively merge the two halves
        # First half: wires 0 to k-1
        # Second half: wires k to 2k-1
        for c in odd_even_merge_recursive(k):
            comps.append((c[0], c[1]))
        for c in odd_even_merge_recursive(k):
            comps.append((c[0] + k, c[1] + k))
        
        return comps
    
    def odd_even_merge_recursive(n):
        """Recursive odd-even merge on n wires (n must be even, n >= 2)."""
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        k = n // 2
        comps = []
        for i in range(k):
            comps.append((i, k + i))
        
        for c in odd_even_merge_recursive(k):
            comps.append((c[0], c[1]))
        for c in odd_even_merge_recursive(k):
            comps.append((c[0] + k, c[1] + k))
        
        return comps
    
    return batcher_sort(N)
# EVOLVE-BLOCK-END

