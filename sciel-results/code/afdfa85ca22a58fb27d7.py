# EVOLVE-BLOCK-START
import time
import random
import sys

def _compute_2_core(n, edges):
    """
    Computes the 2-core of the graph.
    Returns a set of edges that are in the 2-core.
    """
    # Build adjacency list
    adj = {i: set() for i in range(n)}
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    # Queue of vertices with degree < 2
    from collections import deque
    q = deque()
    in_core = [True] * n
    
    for i in range(n):
        if len(adj[i]) < 2:
            q.append(i)
            in_core[i] = False
            
    while q:
        v = q.popleft()
        for u in list(adj[v]):
            if in_core[u]:
                adj[u].discard(v)
                if len(adj[u]) < 2:
                    in_core[u] = False
                    q.append(u)
        adj[v].clear()
        
    # Reconstruct core edges
    core_edges = set()
    for u in range(n):
        if in_core[u]:
            for v in adj[u]:
                if u < v:
                    core_edges.add((u, v))
                    
    return core_edges

def _find_cycle_random_walk(adj, n, rng, max_walks=20, max_len=15):
    """
    Finds a cycle using random walks.
    Returns the longest cycle found among max_walks attempts, or None.
    """
    if not any(len(v) > 0 for v in adj.values()):
        return None
        
    best_cycle = None
    best_len = 0
    
    # Get vertices with degree >= 2
    candidates = [v for v in range(n) if len(adj[v]) >= 2]
    if not candidates:
        return None
        
    for _ in range(max_walks):
        start = rng.choice(candidates)
        path = [start]
        on_path = {start}
        current = start
        
        # Walk
        for step in range(max_len):
            neighbors = list(adj[current])
            if not neighbors:
                break
            
            # Prefer neighbors not in path to extend
            unvisited = [nb for nb in neighbors if nb not in on_path]
            
            if unvisited:
                next_v = rng.choice(unvisited)
            elif start in neighbors:
                # Can close cycle
                if len(path) >= 3:
                    cycle = path[:]
                    if len(cycle) > best_len:
                        best_len = len(cycle)
                        best_cycle = cycle
                break
            else:
                # Stuck, can't extend, can't close
                break
                
            on_path.add(next_v)
            path.append(next_v)
            current = next_v
            
            # Check if we can close now
            if start in adj[current] and len(path) >= 3:
                cycle = path[:]
                if len(cycle) > best_len:
                    best_len = len(cycle)
                    best_cycle = cycle
                # We could break here to keep it simple, or continue to see if we can go longer?
                # If we continue, we might not close at the end.
                # Let's break to ensure we get a valid cycle if we found one.
                # Actually, if we break, we might miss a longer cycle that goes through start and back?
                # No, a simple cycle cannot pass through start twice.
                # So if we are at a vertex connected to start, we can close.
                # If we don't close, we can't visit start again.
                # So we should close if possible to secure the cycle.
                break

    return best_cycle

def _decompose_graph(n, edges, rng, time_budget=20.0):
    """
    Decomposes the graph into cycles and edges.
    Returns a list of pieces.
    """
    start_time = time.time()
    rem_edges = set(tuple(sorted(e)) for e in edges)
    pieces = []
    
    # 1. Compute 2-core
    core_edges = _compute_2_core(n, rem_edges)
    
    # Edges not in core are single pieces
    non_core_edges = rem_edges - core_edges
    for e in non_core_edges:
        pieces.append(list(e))
        
    # 2. Decompose core
    if core_edges:
        # Build adj for core
        adj = {i: set() for i in range(n)}
        for u, v in core_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_core = set(core_edges)
        
        while remaining_core:
            if time.time() - start_time > time_budget:
                # Time up, dump remaining core edges as single edges
                for e in remaining_core:
                    pieces.append(list(e))
                return pieces
            
            # Find a cycle
            cycle = _find_cycle_random_walk(adj, n, rng, max_walks=50, max_len=20)
            
            if cycle:
                # Remove cycle edges
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i+1) % len(cycle)]
                    e = tuple(sorted((u, v)))
                    if e in remaining_core:
                        remaining_core.discard(e)
                        adj[u].discard(v)
                        adj[v].discard(u)
                pieces.append(cycle)
            else:
                # No cycle found, take an arbitrary edge from core
                if not remaining_core:
                    break
                e = next(iter(remaining_core))
                remaining_core.discard(e)
                u, v = e
                adj[u].discard(v)
                adj[v].discard(u)
                pieces.append(list(e))
                
    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    total_start = time.time()
    total_time_limit = 100.0 # Leave 20s buffer
    
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
        
        # Run decomposition
        # We can run multiple attempts with different RNGs and keep the best
        best_pieces = None
        best_count = float('inf')
        
        graph_time_limit = 30.0
        attempts = 5 # Fewer attempts needed as algorithm is faster
        
        for attempt in range(attempts):
            if time.time() - total_start > total_time_limit - 5.0:
                break
            
            rng_decomp = random.Random(seed * 1000 + attempt)
            t_start = time.time()
            pieces = _decompose_graph(n, edges, rng_decomp, time_budget=graph_time_limit / attempts)
            
            # Validate
            if not pieces:
                continue
                
            used_edges = set()
            valid = True
            for p in pieces:
                if len(p) == 2:
                    e = tuple(sorted(p))
                    if e in used_edges or e not in edges:
                        valid = False
                        break
                    used_edges.add(e)
                else:
                    if len(p) < 3 or len(set(p)) != len(p):
                        valid = False
                        break
                    for i in range(len(p)):
                        u = p[i]
                        v = p[(i+1) % len(p)]
                        e = tuple(sorted((u, v)))
                        if e in used_edges or e not in edges:
                            valid = False
                            break
                        used_edges.add(e)
                    if not valid:
                        break
            
            if valid and used_edges == edges:
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
