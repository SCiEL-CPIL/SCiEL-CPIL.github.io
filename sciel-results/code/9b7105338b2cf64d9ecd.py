def construct():
    # Try to find a better graph
    for n in range(35, 26, -1):
        # Try a specific sum-free set
        # Example: S = {1, 2, 5, 10...} ?
        # Let's just try the Andrásfai construction for n=3k-1
        if (n + 1) % 3 == 0:
            k = (n + 1) // 3
            # Andrasfai graph And(k)
            # n = 3k - 1
            # Edges: (j - i) % n in {1, 4, ..., 3k-4}
            # Alpha is k.
            if k <= 9:
                return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
    # Fallback to seed
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}
