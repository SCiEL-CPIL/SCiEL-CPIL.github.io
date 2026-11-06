import time

def _precompute_triples(N=180):
    """Precompute conflicts: for each x, pairs (b,c) such that {x,b,c} is forbidden."""
    conflicts = [set() for _ in range(N+1)]
    
    for x in range(1, N+1):
        x2 = x * x
        # (b-x)(c-x) = x^2, with b > x, c > x, b != c
        for d in range(1, int(x2**0.5) + 1):
            if x2 % d == 0:
                d1 = d
                d2 = x2 // d
                if d1 == d2:
                    continue  # b = c
                b1 = x + d1
                c1 = x + d2
                b2 = x + d2
                c2 = x + d1
                # Check both orderings
                if b1 <= N and c1 <= N and b1 != c1:
                    pair = (min(b1, c1), max(b1, c1))
                    conflicts[x].add(pair)
                if b2 <= N and c2 <= N and b2 != c2:
                    pair = (min(b2, c2), max(b2, c2))
                    conflicts[x].add(pair)
    
    return conflicts

def _greedy(conflicts, order, deadline):
    """Greedily add elements in given order, checking triple conflicts."""
    selected = set()
    result = []
    for x in order:
        if time.time() > deadline:
            break
        if x in selected:
            continue
        # Check if adding x creates a forbidden triple with any two in selected
        can_add = True
        for (b, c) in conflicts[x]:
            if b in selected and c in selected:
                can_add = False
                break
        if can_add:
            result.append(x)
            selected.add(x)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    N = 180
    conflicts = _precompute_triples(N)
    
    if time.time() > deadline:
        return list(range(91, 181))
    
    best = list(range(91, 181))
    best_size = len(best)
    
    # Strategy 1: Large to small
    result = _greedy(conflicts, range(N, 0, -1), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 2: Small to large
    result = _greedy(conflicts, range(1, N+1), deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 3: By conflict count (ascending)
    conflict_counts = [len(conflicts[x]) for x in range(N+1)]
    order = sorted(range(1, N+1), key=lambda x: conflict_counts[x])
    result = _greedy(conflicts, order, deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    # Strategy 4: Random orderings
    import random
    rng = random.Random(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        order = list(range(1, N+1))
        rng.shuffle(order)
        result = _greedy(conflicts, order, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    return best

