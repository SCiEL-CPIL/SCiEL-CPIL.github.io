def construct():
    from math import gcd
    import time
    
    deadline = time.time() + 80  # safety margin
    
    n = 512
    # Precompute compatibility: compat[a][b] = lcm(a,b) <= n
    # Use bitsets for efficiency
    compat = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a, n + 1):
            if a * b // gcd(a, b) <= n:
                compat[a] |= (1 << b)
                compat[b] |= (1 << a)
    
    # Greedy: sort by degree (number of compatible partners), add in that order
    degrees = [bin(compat[i]).count('1') for i in range(1, n + 1)]
    order = sorted(range(1, n + 1), key=lambda x: -degrees[x])
    
    selected = set()
    for x in order:
        if time.time() > deadline - 5:
            break
        ok = True
        for y in selected:
            if not (compat[x] >> y) & 1:
                ok = False
                break
        if ok:
            selected.add(x)
    
    best = list(selected)
    
    # Local search: try to add any element not in set
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for x in range(1, n + 1):
            if x in selected:
                continue
            ok = True
            for y in selected:
                if not (compat[x] >> y) & 1:
                    ok = False
                    break
            if ok:
                selected.add(x)
                improved = True
                break  # restart scan
    
    # Local search: try swaps - remove one, add one or more
    for _ in range(200):
        if time.time() > deadline - 5:
            break
        improved = False
        # Try adding
        for x in range(1, n + 1):
            if x in selected:
                continue
            ok = True
            for y in selected:
                if not (compat[x] >> y) & 1:
                    ok = False
                    break
            if ok:
                selected.add(x)
                improved = True
                break
        if improved:
            continue
        # Try swap: remove one element, see if we can add two
        for y in list(selected):
            if time.time() > deadline - 5:
                break
            selected.remove(y)
            added = 0
            for x in range(1, n + 1):
                if x in selected:
                    continue
                ok = True
                for z in selected:
                    if not (compat[x] >> z) & 1:
                        ok = False
                        break
                if ok:
                    selected.add(x)
                    added += 1
                    break
            if added > 0:
                improved = True
            else:
                selected.add(y)  # put back
        if not improved:
            break
    
    return list(selected)
