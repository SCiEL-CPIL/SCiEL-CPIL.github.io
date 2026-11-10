# EVOLVE-BLOCK-START
import time
import random
import sys

def _find_cycle_greedy(adj, n, rng, time_limit=0.3):
    """
    Finds a long simple cycle using a greedy walk approach.
    Repeatedly tries to extend a path, preferring neighbors that can close the cycle.
    """
    if not adj:
        return None
    
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    best_cycle = None
    best_len = 2
    start_time = time.time()
    
    # Sort candidates by degree (prefer high-degree starts)
    candidates.sort(key=lambda v: len(adj[v]), reverse=True)
    
    # Try a few starts
    for start in rng.sample(candidates, min(8, len(candidates))):
        if time.time() - start_time > time_limit:
            break
        
        path = [start]
        on_path = {start}
        current = start
        
        # Walk: at each step, prefer neighbors not on path, with preference for closing
        while True:
            if time.time() - start_time > time_limit:
                break
            
            neighbors = list(adj[current])
            # Separate: neighbors on path (can close), neighbors not on path (can extend)
            closeable = [nb for nb in neighbors if nb == start and len(path) >= 3]
            extendable = [nb for nb in neighbors if nb not in on_path]
            
            if closeable and not extendable:
                # Must close
                cur_len = len(path)
                if cur_len >= 3 and cur_len > best_len:
                    best_len = cur_len
                    best_cycle = path[:]
                break
            
            if closeable and extendable:
                # Decide: close now or extend?
                # If current length is already good, close. Otherwise extend.
                if len(path) >= 5 or rng.random() < 0.3:
                    cur_len = len(path)
                    if cur_len >= 3 and cur_len > best_len:
                        best_len = cur_len
                        best_cycle = path[:]
                    break
                else:
                    # Extend
                    nb = rng.choice(extendable)
                    on_path.add(nb)
                    path.append(nb)
                    current = nb
                    continue
            
            if extendable:
                # Just extend
                # Prefer neighbors with higher degree (to stay in dense region)
                extendable.sort(key=lambda x: len(adj[x]), reverse=True)
                # Pick from top few
                k = min(3, len(extendable))
                nb = rng.choice(extendable[:k])
                on_path.add(nb)
                path.append(nb)
                current = nb
                continue
            
            # Dead end
            break
    
    return best_cycle

def _decompose_graph_v2(n, edges, rng, time_budget=15.0):
    """
    Decomposes the graph into cycles and edges using greedy cycle finding.
    """
    start_time = time.time()
    rem_edges = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # Build initial adjacency
    adj = {i: set() for i in range(n)}
    for u, v in rem_edges:
        adj[u].add(v)
        adj[v].add(u)
    
    while rem_edges:
        if time.time() - start_time > time_budget:
            # Time up, dump remaining as edges
            for e in rem_edges:
                pieces.append(list(e))
            return pieces
        
        # Find a cycle
        cycle = _find_cycle_greedy(adj, n, rng, time_limit=0.1)
        
        if cycle and len(cycle) >= 3:
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i+1) % len(cycle)]
                e = tuple(sorted((u, v)))
                if e in rem_edges:
                    rem_edges.discard(e)
                    adj[u].discard(v)
                    adj[v].discard(u)
            pieces.append(cycle)
        else:
            # No cycle found, take an arbitrary edge
            if not rem_edges:
                break
            e = next(iter(rem_edges))
            rem_edges.discard(e)
            u, v = e
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append(list(e))
            
    return pieces

def _validate_pieces(pieces, edges):
    """Validate that pieces are edge-disjoint and cover all edges."""
    used_edges = set()
    for p in pieces:
        if len(p) == 2:
            e = tuple(sorted(p))
            if e in used_edges or e not in edges:
                return False
            used_edges.add(e)
        else:
            if len(p) < 3 or len(set(p)) != len(p):
                return False
            for i in range(len(p)):
                u = p[i]
                v = p[(i+1) % len(p)]
                e = tuple(sorted((u, v)))
                if e in used_edges or e not in edges:
                    return False
                used_edges.add(e)
    return used_edges == edges

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_start = time.time()
    total_time_limit = 100.0  # Leave 20s buffer
    
    for n, p, seed in specs:
        # Generate graph
        rng_gen = random.Random(seed)
        order = list(range(n))
        rng_gen.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        for u in range(n):
            for v in range(u+1, n):
                if rng_gen.random() < p:
                    edges.add((u, v))
        
        # Run multiple decomposition attempts
        best_pieces = None
        best_count = float('inf')
        
        graph_time_limit = 30.0
        num_attempts = 15
        
        for attempt in range(num_attempts):
            if time.time() - total_start > total_time_limit - 5.0:
                break
            
            rng_decomp = random.Random(seed * 1000 + attempt * 7 + 42)
            t_start = time.time()
            budget = graph_time_limit / num_attempts
            pieces = _decompose_graph_v2(n, edges, rng_decomp, time_budget=budget)
            
            if not pieces:
                continue
            
            if _validate_pieces(pieces, edges):
                count = len(pieces)
                if count < best_count:
                    best_count = count
                    best_pieces = pieces
        
        if best_pieces is None:
            # Fallback
            best_pieces = [list(e) for e in edges]
        
        results.append(best_pieces)
    
    return results
# EVOLVE-BLOCK-END
