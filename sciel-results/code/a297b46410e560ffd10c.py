import time
import random

N = 15
DEADLINE = 85.0  # 5s safety margin under the 90s wall-clock limit

def _adjacency(edges):
    adj = [set() for _ in range(N)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _degrees(adj):
    return [len(adj[i]) for i in range(N)]

def _is_valid(adj, deg):
    for u in range(N):
        au = adj[u]
        for v in au:
            if v <= u:
                continue
            av = adj[v]
            for s in au:
                if s == v:
                    continue
                ds = deg[s]
                for t in av:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
    return True

def _valid(edges):
    adj = _adjacency(edges)
    deg = _degrees(adj)
    return _is_valid(adj, deg)

def _greedy(order, start_edges):
    edges = list(start_edges)
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    for (u, v) in order:
        key = (min(u, v), max(u, v))
        if key in present:
            continue
        trial = edges + [(u, v)]
        if _valid(trial):
            edges = trial
            present.add(key)
    return edges

def _greedy_degree_aware(edges, all_edges):
    """Greedy that prefers edges connecting vertices with similar degrees (to keep degree diversity)."""
    present = set()
    for u, v in edges:
        present.add((min(u, v), max(u, v)))
    adj = _adjacency(edges)
    deg = _degrees(adj)
    
    # Sort remaining edges by a heuristic: prefer edges that maintain degree diversity
    remaining = [(u, v) for (u, v) in all_edges if (min(u,v), max(u,v)) not in present]
    
    # Try multiple orderings
    best = edges
    best_len = len(edges)
    
    # Ordering 1: by sum of degrees (prefer connecting low-degree vertices)
    order1 = sorted(remaining, key=lambda e: deg[e[0]] + deg[e[1]])
    cand = _greedy(order1, edges)
    if len(cand) > best_len:
        best = cand
        best_len = len(cand)
        present = set((min(u,v), max(u,v)) for u,v in best)
        adj = _adjacency(best)
        deg = _degrees(adj)
        remaining = [(u, v) for (u, v) in all_edges if (min(u,v), max(u,v)) not in present]
    
    # Ordering 2: by difference of degrees (prefer connecting vertices with different degrees)
    order2 = sorted(remaining, key=lambda e: abs(deg[e[0]] - deg[e[1]]), reverse=True)
    cand = _greedy(order2, edges)
    if len(cand) > best_len:
        best = cand
        best_len = len(cand)
    
    return best

def _local_search(base, all_edges, rng, time_limit):
    """Aggressive local search: try edge swaps and additions."""
    best = list(base)
    best_len = len(best)
    t_start = time.time()
    
    improved = True
    while improved and time.time() - t_start < time_limit:
        improved = False
        base = best
        n_base = len(base)
        
        for idx in range(n_base):
            if time.time() - t_start > time_limit:
                break
            
            # Remove one edge
            reduced = base[:idx] + base[idx+1:]
            reduced_set = set((min(u,v), max(u,v)) for u,v in reduced)
            
            # Try adding various edges
            candidates = [(u,v) for (u,v) in all_edges if (min(u,v), max(u,v)) not in reduced_set]
            rng.shuffle(candidates)
            
            # Try up to 30 candidate additions
            for new_edge in candidates[:30]:
                if time.time() - t_start > time_limit:
                    break
                trial = reduced + [new_edge]
                if _valid(trial) and len(trial) > best_len:
                    best = trial
                    best_len = len(trial)
                    improved = True
                    break
            
            # Also try just re-greedy from reduced
            if time.time() - t_start < time_limit:
                order = all_edges[:]
                rng.shuffle(order)
                cand = _greedy(order, reduced)
                if len(cand) > best_len:
                    best = cand
                    best_len = len(cand)
                    improved = True
    
    return best

def construct():
    t0 = time.time()
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    rng = random.Random(20240607)
    
    # Multiple starting points
    starts = [
        [(0, v) for v in range(1, N)],  # star
        [],  # empty
        [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,10), (10,11), (11,12), (12,13), (13,14)],  # path
    ]
    
    best = starts[0]
    best_len = len(best)
    
    def consider(cand):
        nonlocal best, best_len
        if len(cand) > best_len:
            best_len = len(cand)
            best = cand
    
    # Phase 1: Randomized greedy from multiple starts with different orderings
    while time.time() - t0 < DEADLINE * 0.4:
        for start in starts:
            if time.time() - t0 > DEADLINE * 0.4:
                break
            # Multiple random orderings
            for _ in range(3):
                if time.time() - t0 > DEADLINE * 0.4:
                    break
                order = all_edges[:]
                rng.shuffle(order)
                cand = _greedy(order, start)
                consider(cand)
                # Second pass
                if time.time() - t0 < DEADLINE * 0.4:
                    order2 = all_edges[:]
                    rng.shuffle(order2)
                    cand2 = _greedy(order2, cand)
                    consider(cand2)
        
        # Degree-aware greedy from best so far
        if time.time() - t0 < DEADLINE * 0.4:
            cand = _greedy_degree_aware(best, all_edges)
            consider(cand)
    
    # Phase 2: Local search with edge swaps
    while time.time() - t0 < DEADLINE * 0.7:
        time_remaining = DEADLINE * 0.7 - (time.time() - t0)
        if time_remaining < 1.0:
            break
        # Allocate proportional time to each local search iteration
        cand = _local_search(best, all_edges, rng, min(time_remaining, 2.0))
        consider(cand)
    
    # Phase 3: Final greedy passes from the best
    while time.time() - t0 < DEADLINE:
        order = all_edges[:]
        rng.shuffle(order)
        cand = _greedy(order, best)
        consider(cand)
        if time.time() - t0 > DEADLINE:
            break
        order2 = all_edges[:]
        rng.shuffle(order2)
        cand2 = _greedy(order2, cand)
        consider(cand2)
    
    # Exact type conversion: Python int inside a list of lists.
    result = [[int(u), int(v)] for (u, v) in best]
    return result
