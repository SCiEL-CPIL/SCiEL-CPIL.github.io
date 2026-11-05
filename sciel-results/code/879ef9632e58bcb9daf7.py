"""Path decomposition for Gallai's conjecture instances via greedy path extension."""
import time
from collections import defaultdict
from random import Random

def instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def decompose_paths(edges, deadline):
    """Decompose edges into edge-disjoint simple paths, minimizing path count."""
    if not edges:
        return []
    
    # Build adjacency structure
    adj = defaultdict(list)
    edge_set = set()
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].append(v)
        adj[v].append(u)
    
    # Compute degrees
    degree = {u: len(adj[u]) for u in adj}
    
    # Find odd-degree vertices
    odd_vertices = sorted([u for u in degree if degree[u] % 2 == 1])
    odd_count = len(odd_vertices)
    
    # Minimum number of paths needed
    min_paths = max(1, odd_count // 2)
    
    # Strategy: We'll find min_paths paths that cover all edges
    # Approach: 
    # 1. Pair up odd vertices
    # 2. For each pair, find a path between them
    # 3. Remove those edges
    # 4. The remaining graph has all even degrees (or is empty)
    # 5. Decompose remaining into cycles, then break each cycle into a path
    # 6. Merge all these paths together if possible
    
    # Actually, a simpler approach:
    # Use a greedy algorithm that builds paths by extending them
    
    # Let's use a different strategy:
    # Start with each edge as a path
    # Repeatedly merge two paths if they share an endpoint and the connecting edge exists
    
    # Better: Use the fact that we need exactly min_paths paths
    # We can find these by:
    # 1. If odd_count == 0: find an Eulerian circuit, break it into one path
    # 2. If odd_count > 0: pair odd vertices, find paths between pairs, 
    #    then the remaining edges form an Eulerian subgraph
    
    # For now, let's use a practical greedy approach:
    
    # Build path representation
    # Each path: list of vertices
    # Track used edges
    
    remaining_edges = set(edge_set)
    paths = []
    
    # Find a starting vertex (prefer odd vertices)
    start_vertices = odd_vertices if odd_vertices else list(adj.keys())
    
    # We need to find min_paths paths
    # Greedy: repeatedly find a long path, remove its edges
    
    for _ in range(min_paths):
        if not remaining_edges:
            break
        
        # Find a starting vertex
        # Prefer vertices with high degree in remaining graph
        remaining_degree = defaultdict(int)
        for u, v in remaining_edges:
            remaining_degree[u] += 1
            remaining_degree[v] += 1
        
        # Choose start: prefer odd vertices in remaining graph
        rem_odd = [u for u in remaining_degree if remaining_degree[u] % 2 == 1]
        if rem_odd:
            # Pick the one with highest degree
            start = max(rem_odd, key=lambda u: remaining_degree[u])
        else:
            # All even, pick any
            start = max(remaining_degree.keys(), key=lambda u: remaining_degree[u])
        
        # Find a long path starting from start
        path = [start]
        visited_vertices_in_path = {start}
        current = start
        
        while True:
            # Find a neighbor of current that has an unused edge
            found_next = False
            # Sort neighbors by degree (greedy: go to high degree first)
            neighbors = sorted(adj[current], key=lambda v: remaining_degree.get(v, 0), reverse=True)
            
            for nb in neighbors:
                key = (min(current, nb), max(current, nb))
                if key in remaining_edges:
                    # Check if nb is already in path (would create cycle, not simple)
                    if nb not in visited_vertices_in_path:
                        remaining_edges.remove(key)
                        path.append(nb)
                        visited_vertices_in_path.add(nb)
                        current = nb
                        found_next = True
                        break
            
            if not found_next:
                break
        
        if len(path) > 1:
            paths.append(path)
    
    # If there are still remaining edges, we need more paths
    # This shouldn't happen if min_paths is correct, but let's handle it
    while remaining_edges and time.time() < deadline:
        # Find a path in remaining edges
        start = next(iter(remaining_edges))[0]
        path = [start]
        visited_vertices_in_path = {start}
        current = start
        
        while True:
            found_next = False
            neighbors = sorted(adj[current], key=lambda v: remaining_degree.get(v, 0), reverse=True)
            
            for nb in neighbors:
                key = (min(current, nb), max(current, nb))
                if key in remaining_edges and nb not in visited_vertices_in_path:
                    remaining_edges.remove(key)
                    path.append(nb)
                    visited_vertices_in_path.add(nb)
                    current = nb
                    found_next = True
                    break
            
            if not found_next:
                break
        
        if len(path) > 1:
            paths.append(path)
        else:
            # Single edge
            for e in list(remaining_edges):
                u, v = e
                paths.append([u, v])
                remaining_edges.remove(e)
                break
    
    return paths

def construct():
    deadline = time.time() + 100  # Leave safety margin for 120s limit
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)
    
    return results
