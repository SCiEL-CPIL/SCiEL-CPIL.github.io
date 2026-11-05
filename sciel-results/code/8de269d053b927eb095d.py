        def merge_sort_network(n):
            if n <= 1: return []
            mid = n // 2
            left = merge_sort_network(mid)
            right = merge_sort_network(n - mid)
            # Shift right network
            right = [(i+mid, j+mid) for i,j in right]
            # Merge
            merge = merge_network(mid, n - mid)
            return left + right + merge
        
        def merge_network(a, b):
            # Merges two sorted arrays of size a and b
            # Standard merge network construction
            # This is complex.
        