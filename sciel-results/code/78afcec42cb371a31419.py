"""Rewritten program: Maximum Independent Set with B&B and local search."""
import time
import math

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 75.0
    
    N = 256
    nums = list(range(1, N + 1))
    
    # Squares in [2, 512]
    squares = set()
    s = 2
    while s * s <= 2 * N:
        squares.add(s * s)
        s += 1
    
    # Self-conflicts: a where 2a is a perfect square => a = 2*m^2
    self_conflict = set()
    m = 1
    while 2 * m * m <= N:
        self_conflict.add(2 * m * m)
        m += 1
    
    available = [x for x in nums if x not in self_conflict]
    n = len(available)
    idx_map = {x: i for i, x in enumerate(available)}
    
    # Build adjacency
    adj = [set() for _ in range(n)]
    for i in range(n):
        a = available[i]
        for sq in squares:
            b = sq - a
            if b > a and 1 <= b <= N and b in idx_map:
                j = idx_map[b]
                adj[i].add(j)
                adj[j].add(i)
    
    # B&B for Maximum Independent Set
    # Order by degree descending
    degrees = [len(adj[i]) for i in range(n)]
    order = sorted(range(n), key=lambda i: -degrees[i])
    pos = [0] * n
    for p, v in enumerate(order):
        pos[v] = p
    
    # For each position p, neighbors at positions > p
    fwd_adj = [set() for _ in range(n)]
    for i in range(n):
        for j in adj[i]:
            pi, pj = pos[i], pos[j]
            if pi < pj:
                fwd_adj[pi].add(pj)
    
    best_size = 0
    best_choice = [False] * n
    excluded = [False] * n
    chosen = [False] * n
    
    def bb(p, size, ub):
        nonlocal best_size, best_choice
        if time.time() > deadline:
            return
        if p >= n:
            if size > best_size:
                best_size = size
                best_choice = chosen[:]
            return
        if size + ub <= best_size:
            return
        
        # Try include
        if not excluded[p]:
            saved = []
            for nb in fwd_adj[p]:
                if not excluded[nb]:
                    excluded[nb] = True
                    saved.append(nb)
            chosen[p] = True
            bb(p + 1, size + 1, ub - 1)
            chosen[p] = False
            for nb in saved:
                excluded[nb] = False
        
        # Try exclude
        excluded[p] = True
        bb(p + 1, size, ub - 1)
        excluded[p] = False
    
    bb(0, 0, n)
    
    # Build result
    result = [available[order[p]] for p in range(n) if best_choice[p]]
    
    if not result:
        for x in range(1, N + 1):
            if 2 * x not in squares:
                return [x]
        return [1]
    
    # Local search polish
    if time.time() < deadline - 3:
        result_set = set(result)
        in_set = [False] * n
        for v in result_set:
            in_set[idx_map[v]] = True
        
        # Greedy add
        changed = True
        while changed and time.time() < deadline - 3:
            changed = False
            for i in range(n):
                if in_set[i]:
                    continue
                ok = True
                for nb in adj[i]:
                    if in_set[nb]:
                        ok = False
                        break
                if ok:
                    in_set[i] = True
                    changed = True
                    break
        
        # Simple swap improvement
        for i in range(n):
            if not in_set[i]:
                continue
            # Try removing i
            in_set[i] = False
            added = []
            for j in range(n):
                if in_set[j]:
                    continue
                ok = True
                for nb in adj[j]:
                    if in_set[nb]:
                        ok = False
                        break
                if ok:
                    in_set[j] = True
                    added.append(j)
            if len(added) > 1:
                pass  # net gain
            else:
                for j in added:
                    in_set[j] = False
                in_set[i] = True
        
        result = [available[i] for i in range(n) if in_set[i]]
    
    if not result:
        return [1]
    return result

# EVOLVE-BLOCK-END
