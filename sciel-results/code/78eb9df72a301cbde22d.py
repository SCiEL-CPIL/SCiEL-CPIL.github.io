N = 18

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs a sorting network for 18 wires using Batcher's Odd-Even Merge Sort.
    This approach recursively sorts halves and merges them by sorting even and odd 
    indexed subsequences, then comparing adjacent pairs.
    """
    comparators = []
    
    def oems_sort(n, offset):
        """
        Recursively sort n wires starting at offset using Odd-Even Merge Sort.
        """
        if n <= 1:
            return
        
        n1 = n // 2
        n2 = n - n1
        
        # Sort the two halves
        oems_sort(n1, offset)
        oems_sort(n2, offset + n1)
        
        # Merge the two sorted halves
        # The wires are at offset, ..., offset + n - 1
        # We need to merge the first n1 wires and the next n2 wires.
        oems_merge(n1, n2, offset)

    def oems_merge(n1, n2, offset):
        """
        Merge two sorted sequences of lengths n1 and n2 located at offset.
        Uses recursive logic on even and odd indexed elements of the combined sequence.
        """
        N = n1 + n2
        if N <= 1:
            return
        
        if N == 2:
            comparators.append((offset, offset + 1))
            return
        
        # Get the list of wire indices for the combined sequence
        indices = [offset + i for i in range(N)]
        
        # Separate into even and odd logical indices
        even_indices = [indices[i] for i in range(0, N, 2)]
        odd_indices = [indices[i] for i in range(1, N, 2)]
        
        # We need to determine the split sizes for the recursive merge calls.
        # The even_indices list contains elements from the first half and second half.
        # We need to know how many elements from the first half (size n1) are in even_indices
        # and how many are in odd_indices.
        
        # An element at logical index k (0-based) in the combined sequence comes from:
        # - First half if k < n1
        # - Second half if k >= n1
        
        # Count elements from first half in even_indices
        n1_even = 0
        for i in range(0, N, 2):
            if i < n1:
                n1_even += 1
            else:
                break # Since i is increasing, once i >= n1, all subsequent i are >= n1
        
        # Wait, the loop above is not quite right if n1 is not aligned with even indices.
        # Let's just count properly.
        n1_in_even = sum(1 for i in range(0, N, 2) if i < n1)
        n2_in_even = len(even_indices) - n1_in_even
        
        n1_in_odd = sum(1 for i in range(1, N, 2) if i < n1)
        n2_in_odd = len(odd_indices) - n1_in_odd
        
        # Recursively merge the even and odd subsequences
        # Note: The recursive call expects the indices to be contiguous in the *logical* sense,
        # but here we are passing a list of physical wires.
        # We need a helper that takes a list of wires.
        
        oems_merge_list(even_indices, n1_in_even, n2_in_even)
        oems_merge_list(odd_indices, n1_in_odd, n2_in_odd)
        
        # Compare adjacent pairs in the original combined sequence
        for i in range(0, N - 1, 2):
            comparators.append((indices[i], indices[i + 1]))

    def oems_merge_list(indices, n1, n2):
        """
        Merge two sorted sequences of lengths n1 and n2 given by the list of wire indices.
        """
        N = n1 + n2
        if N <= 1:
            return
        
        if N == 2:
            comparators.append((indices[0], indices[1]))
            return
        
        even_indices = [indices[i] for i in range(0, N, 2)]
        odd_indices = [indices[i] for i in range(1, N, 2)]
        
        n1_in_even = sum(1 for i in range(0, N, 2) if i < n1)
        n2_in_even = len(even_indices) - n1_in_even
        
        n1_in_odd = sum(1 for i in range(1, N, 2) if i < n1)
        n2_in_odd = len(odd_indices) - n1_in_odd
        
        oems_merge_list(even_indices, n1_in_even, n2_in_even)
        oems_merge_list(odd_indices, n1_in_odd, n2_in_odd)
        
        for i in range(0, N - 1, 2):
            comparators.append((indices[i], indices[i + 1]))

    oems_sort(N, 0)
    
    return comparators
# EVOLVE-BLOCK-END
