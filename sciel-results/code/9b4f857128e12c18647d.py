def _largest_sidon(arr):
    n = len(arr)
    # Precompute all pairs (i, j) with i <= j
    pairs = []
    for i in range(n):
        for j in range(i, n):
            pairs.append((i, j, arr[i] + arr[j]))
    
    # Check if a subset (bitmask) is a valid Sidon set
    def is_valid(mask):
        used_sums = set()
        for i in range(n):
            if mask & (1 << i):
                for j in range(i, n):
                    if mask & (1 << j):
                        s = arr[i] + arr[j]
                        if s in used_sums:
                            return False
                        used_sums.add(s)
        return True

    # Iterate by size descending
    for size in range(n, 0, -1):
        # Generate all combinations of 'size' indices
        # Using itertools.combinations is efficient enough for n=12
        from itertools import combinations
        for combo in combinations(range(n), size):
            mask = 0
            for idx in combo:
                mask |= (1 << idx)
            if is_valid(mask):
                return size
    return 0
