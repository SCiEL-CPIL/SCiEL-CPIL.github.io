"""Path decomposition for Gallai's conjecture instances via greedy path extension."""
import time
from random import Random

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_paths(edges, deadline):
    """Decompose edges into edge-disjoint simple paths, minimizing path count via greedy extension."""
    if not edges:
        return []
    
    # Build adjacency structure
    adj = {}
    edge_set = set()
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    
    # Sort neighbors for determinism
    for v in adj:
        adj[v].sort()
    
    def find_path(start, remaining_edges, visited_in_path):
        """Try to extend a path from start, using only remaining edges."""
        path = [start]
        visited = {start}
        current = start
        
        while True:
            # Find best next vertex: prefer one that leads to longest extension
            best_next = None
            best_len = 0
            for nb in adj[current]:
                key = (min(current, nb), max(current, nb))
                if key in remaining_edges and nb not in visited:
                    # Try to see how far we can go from nb
                    # For now, just pick the first valid one; we'll do backtracking if needed
                    if best_next is None:
                        best_next = nb
                        best_len = 1
                    # Simple heuristic: prefer neighbors with lower remaining degree
                    # Actually, let's just pick the first valid neighbor for speed
                    break
            
            if best_next is None:
                break
            
            key = (min(current, best_next), max(current, best_next))
            remaining_edges.discard(key)
            visited.add(best_next)
            path.append(best_next)
            current = best_next
        
        return path
    
    def greedy_decompose(remaining_edges, adj_copy):
        """Greedy decomposition: repeatedly find longest possible path."""
        paths = []
        rem = set(remaining_edges)
        
        # Build local adjacency from remaining edges
        local_adj = {v: [] for v in adj_copy}
        for u, v in rem:
            local_adj[u].append(v)
            local_adj[v].append(u)
        
        while rem:
            # Find a starting vertex: prefer odd-degree vertices (Eulerian path property)
            # Or just pick any vertex with remaining edges
            start_candidates = []
            for u, v in rem:
                if u not in start_candidates:
                    start_candidates.append(u)
                if v not in start_candidates:
                    start_candidates.append(v)
            
            # Pick the vertex with highest degree in remaining graph
            start = max(start_candidates, key=lambda x: len(local_adj[x]))
            
            # Build path greedily
            path = [start]
            visited = {start}
            current = start
            local_rem = set(rem)
            
            while True:
                # Find next vertex
                next_v = None
                for nb in sorted(local_adj[current]):
                    key = (min(current, nb), max(current, nb))
                    if key in local_rem and nb not in visited:
                        next_v = nb
                        break
                
                if next_v is None:
                    break
                
                key = (min(current, next_v), max(current, next_v))
                local_rem.discard(key)
                visited.add(next_v)
                path.append(next_v)
                current = next_v
            
            paths.append(path)
            rem = local_rem
            
            # Update local_adj
            local_adj = {v: [] for v in adj_copy}
            for u, v in rem:
                local_adj[u].append(v)
                local_adj[v].append(u)
        
        return paths
    
    # Time-limited search with multiple strategies
    best_paths = None
    best_count = float('inf')
    
    start_time = time.time()
    
    # Strategy 1: Simple greedy
    paths1 = greedy_decompose(set(edge_set), adj)
    count1 = len(paths1)
    best_paths = paths1
    best_count = count1
    
    # Strategy 2: Greedy with different start vertex selection
    # Try starting from each vertex and keep best
    if time.time() < deadline - 5:
        for start_v in sorted(adj.keys()):
            if time.time() >= deadline - 5:
                break
            # Run greedy starting from start_v
            rem = set(edge_set)
            local_adj = {v: [] for v in adj}
            for u, v in rem:
                local_adj[u].append(v)
                local_adj[v].append(u)
            
            paths = []
            # First path starts from start_v
            path = [start_v]
            visited = {start_v}
            current = start_v
            local_rem = set(rem)
            
            while True:
                next_v = None
                for nb in sorted(local_adj[current]):
                    key = (min(current, nb), max(current, nb))
                    if key in local_rem and nb not in visited:
                        next_v = nb
                        break
                
                if next_v is None:
                    break
                
                key = (min(current, next_v), max(current, next_v))
                local_rem.discard(key)
                visited.add(next_v)
                path.append(next_v)
                current = next_v
            
            paths.append(path)
            rem = local_rem
            
            # Continue with greedy
            while rem:
                local_adj = {v: [] for v in adj}
                for u, v in rem:
                    local_adj[u].append(v)
                    local_adj[v].append(u)
                
                start_candidates = []
                for u, v in rem:
                    if u not in start_candidates:
                        start_candidates.append(u)
                    if v not in start_candidates:
                        start_candidates.append(v)
                
                if not start_candidates:
                    break
                    
                start = max(start_candidates, key=lambda x: len(local_adj[x]))
                
                path = [start]
                visited = {start}
                current = start
                local_rem = set(rem)
                
                while True:
                    next_v = None
                    for nb in sorted(local_adj[current]):
                        key = (min(current, nb), max(current, nb))
                        if key in local_rem and nb not in visited:
                            next_v = nb
                            break
                    
                    if next_v is None:
                        break
                    
                    key = (min(current, next_v), max(current, next_v))
                    local_rem.discard(key)
                    visited.add(next_v)
                    path.append(next_v)
                    current = next_v
                
                paths.append(path)
                rem = local_rem
            
            count = len(paths)
            if count < best_count:
                best_count = count
                best_paths = paths
    
    # Verify and return
    if best_paths is None:
        # Fallback: each edge as its own path
        return [[list(e) for e in edges]]
    
    return best_paths

def construct():
    """Return three lists of paths, one per fixed instance graph."""
    deadline = time.time() + 120  # Safety margin
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)
        deadline = min(deadline, time.time() + 30)  # Reserve time for remaining instances
    
    return results
