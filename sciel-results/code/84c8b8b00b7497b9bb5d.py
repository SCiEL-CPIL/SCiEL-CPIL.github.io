def _kick_and_refill(S, pool, rng, deadline):
    S = S.copy()
    S_set = set(S)
    pool_set = set(pool)
    
    # Remove 2-5 random elements
    k = rng.randint(2, 5)
    removed = []
    for _ in range(k):
        if not S:
            break
        idx = rng.randint(0, len(S) - 1)
        removed.append(S.pop(idx))
    
    # Add removed elements back to pool
    pool = pool + removed
    
    # Greedily add from pool
    rng.shuffle(pool)
    for x in pool:
        if time.time() > deadline:
            break
        if x in S_set:
            continue
        if _can_add_fast(x, S):
            S.append(x)
            S_set.add(x)
            
    return S
