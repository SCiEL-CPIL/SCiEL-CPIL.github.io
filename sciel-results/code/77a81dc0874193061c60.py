"""Decompose three fixed graphs into cycles and edges, minimizing piece count."""
import time
import random
from collections import defaultdict

_DEADLINE_SECONDS = 145.0

def _instance(n, p, seed):
    """Replicate the verifier's instance generator exactly."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_longest_cycle(adj, remaining_edges, n):
    """
    Find the longest simple cycle in the graph defined by adj and remaining_edges.
    Uses a randomized DFS with backtracking, limited by depth.
    Returns list of vertices or None.
    """
    if not remaining_edges:
        return None
    
    # Heuristic: Start from a vertex with high degree
    degrees = {u: len(adj[u]) for u in adj}
    candidates = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)
    
    best_cycle = None
    best_len = 0
    
    # Try a few random starts and deterministic high-degree starts
    starts = candidates[:5] + random.sample(list(adj.keys()), min(5, len(adj)))
    
    for start in starts:
        if start not in adj or len(adj[start]) < 2:
            continue
        
        # DFS to find a cycle
        path = [start]
        on_path = {start}
        
        def dfs(u, depth):
            nonlocal best_cycle, best_len
            if depth > n:
                return
            # Prune: if max possible length is not better, stop?
            # Hard to estimate. Just limit depth.
            
            for v in sorted(adj[u]):
                edge = (min(u, v), max(u, v))
                if edge not in remaining_edges:
                    continue
                if v == start:
                    if len(path) > best_len:
                        best_len = len(path)
                        best_cycle = path[:]
                    continue
                if v not in on_path:
                    on_path.add(v)
                    path.append(v)
                    dfs(v, depth + 1)
                    path.pop()
                    on_path.discard(v)
        
        dfs(start, 0)
        
    return best_cycle

def _decompose_greedy_longest(edges, n):
    """
    Greedy decomposition: repeatedly find and remove the longest cycle.
    """
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    
    pieces = []
    remaining_edges = set((min(u, v), max(u, v)) for u, v in edges)
    
    while remaining_edges:
        cycle = _find_longest_cycle(adj, remaining_edges, n)
        if cycle:
            pieces.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.discard(edge)
                    adj[u].discard(v)
                    adj[v].discard(u)
        else:
            # No cycles left, take one edge
            edge = next(iter(remaining_edges))
            remaining_edges.discard(edge)
            u, v = edge
            pieces.append([u, v])
            adj[u].discard(v)
            adj[v].discard(u)
            
    return pieces

def _local_reoptimize(cycles, free_edges, adj, n, time_limit):
    """
    Local search step:
    1. Pick a random cycle.
    2. Remove it.
    3. Collect all free edges incident to its vertices + the cycle's edges.
    4. Try to repartition this local set of edges into cycles that yield a better score.
    5. If better, apply the change.
    """
    if not cycles:
        return False, 0
    
    # Pick a random cycle to re-optimize
    idx = random.randint(0, len(cycles) - 1)
    C = cycles[idx]
    
    # Get vertices in C
    C_verts = set(C)
    
    # Collect local edges:
    # 1. Edges of C
    # 2. Free edges where both endpoints are in C_verts
    # 3. Free edges where one endpoint is in C_verts and the other is a neighbor?
    #    To keep it local and simple, let's only consider free edges with BOTH endpoints in C_verts.
    #    This is a "local" improvement. It might miss global improvements but is fast.
    
    local_edges = set()
    for i in range(len(C)):
        u = C[i]
        v = C[(i + 1) % len(C)]
        local_edges.add((min(u, v), max(u, v)))
        
    for e in free_edges:
        u, v = e
        if u in C_verts and v in C_verts:
            local_edges.add(e)
            
    # Remove C from cycles and add its edges to free_edges temporarily
    cycles.pop(idx)
    for e in local_edges:
        free_edges.add(e)
        
    # Now, try to find the best decomposition of local_edges
    # We want to maximize the number of edges covered by cycles in local_edges.
    # Let's use a greedy approach on this small subgraph.
    
    # Build local adj
    local_adj = defaultdict(set)
    for u, v in local_edges:
        local_adj[u].add(v)
        local_adj[v].add(u)
        
    new_cycles = []
    local_remaining = set(local_edges)
    
    # Greedy: find cycles in local_remaining
    while True:
        # Find any cycle in local_remaining
        # Use a simple DFS
        found = None
        for start in list(local_adj.keys()):
            if len(local_adj[start]) < 2:
                continue
            path = [start]
            on_path = {start}
            
            def dfs_local(u, depth):
                nonlocal found
                if found:
                    return
                if depth > n:
                    return
                for v in local_adj[u]:
                    edge = (min(u, v), max(u, v))
                    if edge not in local_remaining:
                        continue
                    if v == start:
                        if len(path) >= 3:
                            found = path[:]
                            return
                        continue
                    if v not in on_path:
                        on_path.add(v)
                        path.append(v)
                        dfs_local(v, depth + 1)
                        path.pop()
                        on_path.discard(v)
                        if found:
                            return
            
            dfs_local(start, 0)
            if found:
                break
                
        if not found:
            break
            
        new_cycles.append(found)
        for i in range(len(found)):
            u = found[i]
            v = found[(i + 1) % len(found)]
            local_remaining.discard((min(u, v), max(u, v)))
            
    # Calculate improvement
    # Old state: 1 cycle C (length L_old), 0 free edges from local_edges (they were in C or free)
    # Wait, the free edges in local_edges were already free.
    # Old pieces count for this local part: 1 (for C) + (number of free edges in local_edges that were NOT in C)
    # Actually, let's just compare the total edges covered by cycles.
    
    old_edges_covered = len(C)
    new_edges_covered = sum(len(c) for c in new_cycles)
    
    # The change is valid if new_edges_covered > old_edges_covered
    # If equal, we might still take it to escape local optima (simulated annealing), 
    # but for now, let's strictly improve or keep equal.
    
    if new_edges_covered > old_edges_covered:
        # Apply change
        cycles.extend(new_cycles)
        # Update free_edges: remove the edges that are now in new_cycles
        for c in new_cycles:
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                free_edges.discard((min(u, v), max(u, v)))
        # The edges in local_edges that are NOT in new_cycles remain in free_edges (they were added before)
        return True, new_edges_covered - old_edges_covered
    else:
        # Revert
        # Remove new_cycles (none added to global yet, but we popped C)
        cycles.insert(idx, C)
        # Remove local_edges from free_edges (they were added)
        for e in local_edges:
            free_edges.discard(e)
        return False, 0

def construct():
    start_time = time.time()
    specs = [(20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)]
    results = []

    for n, p, seed in specs:
        elapsed = time.time() - start_time
        remaining_time = _DEADLINE_SECONDS - elapsed
        if remaining_time < 5:
            edges = _instance(n, p, seed)
            results.append([[u, v] for u, v in edges])
            continue

        edges = _instance(n, p, seed)
        edges_set = set((min(u, v), max(u, v)) for u, v in edges)
        
        # 1. Initial Greedy Decomposition
        pieces = _decompose_greedy_longest(edges, n)
        
        # Separate cycles and edges
        cycles = []
        free_edges = set()
        for p in pieces:
            if len(p) >= 3:
                cycles.append(p)
            else:
                free_edges.add((min(p[0], p[1]), max(p[0], p[1])))
                
        # Build adj for local search
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # 2. Local Search Optimization
        # Run until time limit or no improvements
        improvement_threshold = 0
        last_improvement_time = time.time()
        
        while time.time() - start_time < remaining_time - 5:
            improved, gain = _local_reoptimize(cycles, free_edges, adj, n, remaining_time - (time.time() - start_time))
            if improved:
                last_improvement_time = time.time()
            
        # 3. Final Validation and Formatting
        final_pieces = []
        used_edges = set()
        
        for c in cycles:
            valid = True
            for i in range(len(c)):
                u = c[i]
                v = c[(i + 1) % len(c)]
                e = (min(u, v), max(u, v))
                if e not in edges_set or e in used_edges:
                    valid = False
                    break
                used_edges.add(e)
            if valid:
                final_pieces.append(c)
                
        for e in free_edges:
            if e in edges_set and e not in used_edges:
                final_pieces.append(list(e))
                used_edges.add(e)
                
        # Add any missed edges (shouldn't happen if logic is correct, but safe)
        for e in edges_set:
            if e not in used_edges:
                final_pieces.append(list(e))
                
        results.append(final_pieces)

    return results
