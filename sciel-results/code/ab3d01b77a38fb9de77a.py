def _cartesian_product(adj1, adj2):
    """Compute the Cartesian product of two graphs."""
    n1, n2 = len(adj1), len(adj2)
    n = n1 * n2
    adj = [set() for _ in range(n)]
    for i in range(n1):
        for j in range(n2):
            v = i * n2 + j
            # Connect to (i, j') for j' in adj2[j]
            for j2 in adj2[j]:
                v2 = i * n2 + j2
                adj[v].add(v2)
            # Connect to (i', j) for i' in adj1[i]
            for i2 in adj1[i]:
                v2 = i2 * n2 + j
                adj[v].add(v2)
    return [sorted(s) for s in adj]
