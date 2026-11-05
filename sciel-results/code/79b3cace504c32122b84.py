import time
import random
from collections import defaultdict

def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_path(adj, start, end, visited):
    """Find a simple path from start to end in adj using DFS."""
    if start == end:
        return [start]
    visited.add(start)
    for neighbor in adj[start]:
        if neighbor not in visited:
            path = _find_path(adj, neighbor, end, visited)
            if path:
                return [start] + path
    return None

def _greedy_decompose(edges):
    """
    Initial decomposition using a greedy strategy.
    We prioritize extending from vertices with high remaining degree to create long paths.
    """
    if not edges:
        return []
        
    remaining = set(edges)
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    paths = []
    
    while remaining:
        # Pick a seed edge. 
        # Heuristic: pick an edge incident to a vertex with the highest degree.
        # This helps "consume" high-degree vertices into long paths.
        max_deg = -1
        chosen_edge = None
        for e in remaining:
            u, v = e
            deg_u = len(adj[u])
            deg_v = len(adj[v])
            if deg_u > max_deg or deg_v > max_deg:
                max_deg = max(deg_u, deg_v)
                chosen_edge = e
        
        if chosen_edge is None:
            chosen_edge = next(iter(remaining))
            
        u, v = chosen_edge
        remaining.remove(chosen_edge)
        adj[u].discard(v)
        adj[v].discard(u)
        
        path = [u, v]
        path_set = {u, v}
        
        # Extend from v
        curr = v
        while True:
            # Choose next vertex from adj[curr]
            # Heuristic: pick the one with the highest degree to keep the path "dense"
            candidates = [w for w in adj[curr] if w not in path_set]
            if not candidates:
                break
            # Sort by degree descending
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            next_v = candidates[0]
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.append(next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        # Extend from u
        curr = u
        while True:
            candidates = [w for w in adj[curr] if w not in path_set]
            if not candidates:
                break
            candidates.sort(key=lambda x: len(adj[x]), reverse=True)
            next_v = candidates[0]
            
            e2 = tuple(sorted((curr, next_v)))
            if e2 in remaining:
                remaining.remove(e2)
                adj[curr].discard(next_v)
                adj[next_v].discard(curr)
                path.insert(0, next_v)
                path_set.add(next_v)
                curr = next_v
            else:
                break

        paths.append(path)
        
    return paths

def _merge_paths(paths, edges):
    """
    Try to merge two paths if they are vertex-disjoint and can be connected by an edge.
    Returns a new list of paths if a merge was successful, else the original list.
    """
    n = len(paths)
    if n < 2:
        return paths
    
    # Create a set of all edges for quick lookup
    edge_set = set(edges)
    
    # We want to find a pair (i, j) such that:
    # 1. paths[i] and paths[j] are vertex disjoint
    # 2. There is an edge connecting an endpoint of i to an endpoint of j
    
    # To be efficient, we can just iterate. n is small.
    # We will try to find the "best" merge. What is best?
    # Maybe merging paths that are "close" in terms of endpoints?
    # For now, just find ANY valid merge.
    
    # Precompute vertex sets
    vertex_sets = [set(p) for p in paths]
    
    for i in range(n):
        for j in range(i + 1, n):
            if vertex_sets[i] & vertex_sets[j]:
                continue
                
            p_i = paths[i]
            p_j = paths[j]
            
            # Check 4 possible connections
            # 1. p_i[-1] -- p_j[0]
            if (p_i[-1], p_j[0]) in edge_set or (p_j[0], p_i[-1]) in edge_set:
                new_path = p_i + p_j
                new_paths = paths[:i] + paths[i+1:j] + [new_path] + paths[j+1:]
                return new_paths
            # 2. p_i[-1] -- p_j[-1]
            if (p_i[-1], p_j[-1]) in edge_set or (p_j[-1], p_i[-1]) in edge_set:
                new_path = p_i + p_j[::-1]
                new_paths = paths[:i] + paths[i+1:j] + [new_path] + paths[j+1:]
                return new_paths
            # 3. p_i[0] -- p_j[0]
            if (p_i[0], p_j[0]) in edge_set or (p_j[0], p_i[0]) in edge_set:
                new_path = p_i[::-1] + p_j
                new_paths = paths[:i] + paths[i+1:j] + [new_path] + paths[j+1:]
                return new_paths
            # 4. p_i[0] -- p_j[-1]
            if (p_i[0], p_j[-1]) in edge_set or (p_j[-1], p_i[0]) in edge_set:
                new_path = p_i[::-1] + p_j[::-1]
                new_paths = paths[:i] + paths[i+1:j] + [new_path] + paths[j+1:]
                return new_paths
                
    return paths

def decompose(edges):
    if not edges:
        return []
        
    # 1. Initial greedy decomposition
    paths = _greedy_decompose(edges)
    
    # 2. Iterative merging
    # Loop until no more merges are possible
    while True:
        new_paths = _merge_paths(paths, edges)
        if len(new_paths) == len(paths):
            break
        paths = new_paths
        
    return paths

def construct():
    deadline = time.time() + 140
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose(edges)
        results.append(paths)
        
        if time.time() > deadline:
             # Safety break
             pass
             
    return results
