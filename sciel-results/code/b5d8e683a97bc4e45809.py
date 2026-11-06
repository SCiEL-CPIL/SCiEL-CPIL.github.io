N = 18

# EVOLVE-BLOCK-START
def construct():
    def batcher_merge(n, offset1, offset2):
        # Merges two sorted sequences of length n
        # Seq 1 starts at offset1, Seq 2 starts at offset2
        # Assumes offset2 == offset1 + n
        if n == 1:
            return [(offset1, offset2)]
        
        h = n // 2
        comparators = []
        
        # Recursively merge the lower halves
        # Lower half of Seq 1: offset1 to offset1 + h - 1
        # Lower half of Seq 2: offset2 to offset2 + h - 1
        comparators.extend(batcher_merge(h, offset1, offset2))
        
        # Recursively merge the upper halves
        # Upper half of Seq 1: offset1 + h to offset1 + n - 1
        # Upper half of Seq 2: offset2 + h to offset2 + n - 1
        comparators.extend(batcher_merge(h, offset1 + h, offset2 + h))
        
        # Cross comparisons
        for i in range(h):
            # Compare corresponding elements from the two merged halves
            # 1. i-th of lower1 vs i-th of lower2
            comparators.append((offset1 + i, offset2 + i))
            # 2. i-th of upper1 vs i-th of upper2
            comparators.append((offset1 + h + i, offset2 + h + i))
            # 3. i-th of lower1 vs i-th of upper1
            comparators.append((offset1 + i, offset1 + h + i))
            # 4. i-th of lower2 vs i-th of upper2
            comparators.append((offset2 + i, offset2 + h + i))
            
        return comparators

    def batcher_sort(n):
        if n == 1:
            return []
        
        # Split into two halves
        # We need to sort the first half and the second half
        # Then merge them
        # Note: Batcher's sort is recursive.
        # Sort first half (0 to n/2 - 1)
        # Sort second half (n/2 to n - 1)
        # Then merge them.
        
        # However, the merge function above assumes the two inputs are contiguous blocks of size n.
        # So we need to adapt the indices.
        
        # Let's create a helper that sorts a range [start, end)
        def sort_range(start, length):
            if length == 1:
                return []
            h = length // 2
            left = sort_range(start, h)
            right = sort_range(start + h, length - h)
            # Merge left and right
            # The merge function expects two blocks of size h?
            # No, my batcher_merge expects two blocks of size n to produce 2n.
            # Here we have two blocks of size h to produce 2h.
            merged = batcher_merge(h, start, start + h)
            return left + right + merged

        return sort_range(0, n)

    comparators = batcher_sort(N)
    
    # Remove duplicates
    unique_comparators = list(set(comparators))
    
    # Ensure i < j
    final_comparators = []
    for i, j in unique_comparators:
        if i > j:
            i, j = j, i
        final_comparators.append((i, j))
        
    # Sort for determinism (optional)
    final_comparators.sort()
    
    return final_comparators
# EVOLVE-BLOCK-END
