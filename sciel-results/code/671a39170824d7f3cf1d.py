def batcher_oes(n):
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    # Recursively sort two halves
    left = batcher_oes(n // 2)
    right = [(i + n//2, j + n//2) for (i, j) in left]
    
    # Odd-even merge
    def merge(n):
        if n == 1:
            return [(0, 1)]
        # Compare pairs (0,1), (2,3), ...
        comparators = [(2*i, 2*i+1) for i in range(n)]
        # Recursively merge even and odd subsequences
        even = merge(n)  # wait, this is wrong
        ...
    ...
