# EVOLVE-BLOCK-START
import time
import random
import sys

def _find_cycle_longest(adj, n, rng, time_limit=0.5):
    """
    Finds a long simple cycle in the graph defined by adj (dict of sets).
    Uses randomized DFS with backtracking.
    Returns list of vertices in the cycle or None.
    """
    if not adj:
        return None
    
    # Start from a random vertex with degree >= 2
    candidates = [v for v in adj if len(adj[v]) >= 2]
    if not candidates:
        return None
    
    best_cycle = None
    best_len = 2
    start_time = time.time()
    
    # Try multiple starts
    for start in rng.sample(candidates, min(5, len(candidates))):
        if time.time() - start_time > time_limit:
            break
            
        # DFS
        path = [start]
        on_path = {start}
        
        def dfs(current):
            nonlocal best_cycle, best_len
            if time.time() - start_time > time_limit:
                return
            
            # Check if we can close the cycle
            if start in adj[current]:
                cur_len = len(path)
                if cur_len >= 3 and cur_len > best_len:
                    best_len = cur_len
                    best_cycle = path[:]
            
            # Explore neighbors
            # Heuristic: Sort neighbors by degree (higher degree first to stay in core, or random)
            # Let's use random order but prioritize unvisited
            neighbors = list(adj[current])
            rng.shuffle(neighbors)
            
            for nb in neighbors:
                if nb == start:
                    continue # Handled above, or allow closing? 
                    # If we allow closing here, we update best. 
                    # But we want to explore deeper if possible.
                    # If we close, we don't recurse.
                
                if nb not in on_path:
                    on_path.add(nb)
                    path.append(nb)
                    dfs(nb)
                    path.pop()
                    on_path.discard(nb)
                
                # Early termination if we found a Hamiltonian cycle?
                if best_len == n:
                    return

        dfs(start)
        
        if best_len == n:
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
        cycle = _find_cycle_longest(adj, n, rng, time_limit=0.2)
        
        if cycle:
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
        
        # Run multiple decomposition attempts
        best_pieces = None
        best_count = float('inf')
        
        # Determine time budget per attempt
        # We have ~100s total. 3 graphs. 
        # Let's allocate 30s per graph.
        graph_time_limit = 30.0
        
        # Try different random seeds for the decomposition
        for attempt in range(10):
            if time.time() - total_start > total_time_limit - 5.0:
                break
            
            rng_decomp = random.Random(seed * 1000 + attempt)
            t_start = time.time()
            pieces = _decompose_graph(n, edges, rng_decomp, time_budget=graph_time_limit / 10.0)
            
            # Validate
            if not pieces:
                continue
                
            # Simple validation: check edge coverage and disjointness
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
            
            if time.time() - t_start > graph_time_limit / 10.0:
                # If one attempt took too long, stop trying more for this graph?
                # No, we just break the loop if we hit total time limit.
                pass

        if best_pieces is None:
            # Fallback
            best_pieces = [list(e) for e in edges]
            
        results.append(best_pieces)
        
    return results
# EVOLVE-BLOCK-END
