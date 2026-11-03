def _generate_random_regular(n, k, rng):
    if n * k % 2 != 0 or n < k:
        return None
    # Use a more robust generation
    # 1. Create stubs
    # 2. Shuffle
    # 3. Pair
    # 4. Fix loops and multi-edges by swapping
    
    adj = [set() for _ in range(n)]
    stubs = [i for i in range(n) for _ in range(k)]
    
    for _ in range(100): # Max attempts
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        success = True
        for i in range(0, n * k, 2):
            u, v = stubs[i], stubs[i + 1]
            if u == v or v in adj[u]:
                success = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if success:
            # Verify
            if all(len(lst) == k for lst in adj):
                return [sorted(lst) for lst in adj]
    return None
