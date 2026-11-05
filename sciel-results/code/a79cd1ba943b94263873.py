"""Path decomposition for Gallai's conjecture instances."""
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


def _find_long_simple_path(adj, edge_set, start, deadline):
    """Greedy: find a long simple path starting at start using edges in edge_set."""
    path = [start]
    used_edges = set()
    current = start
    visited = {start}
    
    while time.time() < deadline:
        # Find neighbors with unused edges
        candidates = []
        for nb in adj[current]:
            key = (min(current, nb), max(current, nb))
            if key in edge_set and key not in used_edges:
                if nb not in visited:
                    # Score: prefer neighbors with higher degree in remaining graph
                    deg = sum(1 for u, v in edge_set if (u == nb or v == nb))
                    candidates.append((deg, nb))
        
        if not candidates:
            break
        
        # Pick highest degree
        candidates.sort(reverse=True)
        best_deg, best_nb = candidates[0]
        
        key = (min(current, best_nb), max(current, best_nb))
        used_edges.add(key)
        path.append(best_nb)
        visited.add(best_nb)
        current = best_nb
    
    return path, used_edges


def _try_merge_paths(paths, edge_set, deadline):
    """Try to merge two paths into one if there's a connecting edge between endpoints."""
    if len(paths) < 2:
        return paths
    
    # Track all edges used in current paths
    used_edges = set()
    for path in paths:
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            key = (min(a, b), max(a, b))
            used_edges.add(key)
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(paths)):
            if changed:
                break
            for j in range(i + 1, len(paths)):
                if changed:
                    break
                p1 = paths[i]
                p2 = paths[j]
                
                # Try all endpoint combinations
                merged = None
                for a, b, rev1, rev2 in [
                    (p1[0], p2[0], False, False),
                    (p1[0], p2[-1], False, True),
                    (p1[-1], p2[0], True, False),
                    (p1[-1], p2[-1], True, True),
                ]:
                    key = (min(a, b), max(a, b))
                    if key in edge_set and key not in used_edges:
                        # Check if merged path is simple
                        if rev1:
                            p1m = p1[::-1]
                        else:
                            p1m = p1
                        if rev2:
                            p2m = p2[::-1]
                        else:
                            p2m = p2
                        merged_path = p1m + p2m
                        # Check simple
                        if len(merged_path) == len(set(merged_path)):
                            merged = (merged_path, key)
                            break
                
                if merged:
                    merged_path, edge_key = merged
                    paths[i] = merged_path
                    paths.pop(j)
                    used_edges.add(edge_key)
                    changed = True
                    break
    
    return paths


def decompose_paths(edges, deadline):
    """Decompose edges into edge-disjoint simple paths, minimizing path count."""
    if not edges:
        return []
    
    # Build adjacency and edge set
    adj = defaultdict(set)
    edge_set = set()
    for e in edges:
        u, v = e
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        adj[u].add(v)
        adj[v].add(u)
    
    # Compute degrees
    degree = {u: len(adj[u]) for u in adj}
    
    # Find odd-degree vertices
    odd_vertices = sorted([u for u in degree if degree[u] % 2 == 1])
    odd_count = len(odd_vertices)
    
    # Minimum number of paths needed
    min_paths = max(1, odd_count // 2)
    
    # Strategy:
    # 1. Pair up odd vertices
    # 2. For each pair, find a simple path between them
    # 3. Remove those edges
    # 4. The remaining graph has all even degrees
    # 5. Decompose remaining into cycles
    # 6. Convert cycles to paths and merge
    
    paths = []
    remaining_edges = set(edge_set)
    
    # Pair odd vertices and find paths between them
    if odd_count > 0:
        pairs = []
        for i in range(0, odd_count, 2):
            pairs.append((odd_vertices[i], odd_vertices[i + 1]))
        
        for u, v in pairs:
            # Find a simple path from u to v
            found = False
            for _ in range(5):
                path, used = _find_long_simple_path(adj, remaining_edges, u, deadline)
                if v in path:
                    idx = path.index(v)
                    subpath = path[:idx + 1]
                    sub_edges = set()
                    for i in range(len(subpath) - 1):
                        a, b = subpath[i], subpath[i + 1]
                        key = (min(a, b), max(a, b))
                        sub_edges.add(key)
                    remaining_edges -= sub_edges
                    paths.append(subpath)
                    found = True
                    break
            
            if not found:
                # Fallback: use a single edge from u
                for nb in adj[u]:
                    key = (min(u, nb), max(u, nb))
                    if key in remaining_edges:
                        paths.append([u, nb])
                        remaining_edges.remove(key)
                        found = True
                        break
    
    # Decompose remaining edges (all even degrees) into paths
    while remaining_edges and time.time() < deadline:
        # Find a cycle in remaining_edges
        start = None
        for e in remaining_edges:
            start = e[0]
            break
        if start is None:
            break
        
        # Find a cycle by walking
        cycle = [start]
        visited_vertices = {start}
        current = start
        cycle_edges = set()
        closed = False
        
        while time.time() < deadline:
            found_next = False
            for nb in sorted(adj[current]):
                key = (min(current, nb), max(current, nb))
                if key in remaining_edges and key not in cycle_edges:
                    if nb == start and len(cycle) > 1:
                        cycle_edges.add(key)
                        closed = True
                        break
                    elif nb not in visited_vertices:
                        cycle_edges.add(key)
                        cycle.append(nb)
                        visited_vertices.add(nb)
                        current = nb
                        found_next = True
                        break
            
            if not found_next and not closed:
                break
        
        if closed and len(cycle) > 2:
            # Remove cycle edges from remaining
            for key in cycle_edges:
                if key in remaining_edges:
                    remaining_edges.remove(key)
            
            # Convert cycle to path
            if len(set(cycle)) == len(cycle) - 1:
                path = cycle[:-1]
                paths.append(path)
            else:
                # Find simple path within cycle
                seen = set()
                simple_path = []
                for v in cycle:
                    if v not in seen:
                        seen.add(v)
                        simple_path.append(v)
                if len(simple_path) > 1:
                    valid = True
                    for i in range(len(simple_path) - 1):
                        a, b = simple_path[i], simple_path[i + 1]
                        key = (min(a, b), max(a, b))
                        if key not in cycle_edges:
                            valid = False
                            break
                    if valid:
                        paths.append(simple_path)
                    else:
                        paths.append(cycle[:2])
        elif remaining_edges:
            # No cycle found, take a single edge
            e = next(iter(remaining_edges))
            paths.append([e[0], e[1]])
            remaining_edges.remove(e)
    
    # Try to merge paths to reduce count
    paths = _try_merge_paths(paths, edge_set, deadline)
    
    return paths


def construct():
    deadline = time.time() + 100  # Leave safety margin for 120s limit
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_paths(edges, deadline)
        results.append(paths)
    
    return results
