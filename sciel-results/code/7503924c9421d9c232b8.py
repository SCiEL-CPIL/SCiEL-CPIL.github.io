"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

_DEADLINE = 120.0  # wall-clock seconds

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    # sort neighbors for determinism
    for u in adj:
        adj[u].sort()
    return adj

def _path_to_edges(path):
    """Convert a vertex list path to a set of frozenset edges."""
    edges = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        edges.add(frozenset((a, b)))
    return edges

def _is_valid_decomposition(paths, edges):
    """Check that paths are simple, use graph edges, edge-disjoint, and cover all edges."""
    edge_set = {frozenset(e) for e in edges}
    used = set()
    for path in paths:
        # check simple (no repeated vertices)
        if len(path) != len(set(path)):
            return False
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            e = frozenset((a, b))
            if e not in edge_set:
                return False
            if e in used:
                return False
            used.add(e)
    return used == edge_set

def _greedy_path_cover(edges, adj, time_limit=10.0):
    """
    Greedy approach: repeatedly find the longest path in the remaining graph.
    This gives a valid edge decomposition.
    """
    remaining = set(frozenset(e) for e in edges)
    if not remaining:
        return []
    
    paths = []
    start_time = time.time()
    
    # Build adjacency from remaining edges
    rem_adj = {}
    for e in remaining:
        a, b = tuple(e)
        rem_adj.setdefault(a, set()).add(b)
        rem_adj.setdefault(b, set()).add(a)
    
    while remaining:
        if time.time() - start_time > time_limit:
            break
        
        # Find a vertex with odd degree (if any), otherwise any vertex
        # In a graph where all vertices have even degree, we can start anywhere
        start = None
        for v in sorted(rem_adj.keys()):
            if len(rem_adj[v]) % 2 == 1:
                start = v
                break
        if start is None:
            start = sorted(rem_adj.keys())[0] if rem_adj else None
        if start is None:
            break
        
        # DFS to find a path from start
        path = [start]
        visited_edges = set()
        
        def dfs(node):
            if time.time() - start_time > time_limit:
                return
            # Find an unvisited edge from node
            neighbors = list(rem_adj.get(node, []))
            for nb in neighbors:
                e = frozenset((node, nb))
                if e not in visited_edges:
                    visited_edges.add(e)
                    path.append(nb)
                    dfs(nb)
                    return
        
        dfs(start)
        
        if len(path) >= 2:
            paths.append(path)
            # Remove used edges from remaining
            for e in visited_edges:
                a, b = tuple(e)
                remaining.discard(e)
                if a in rem_adj:
                    rem_adj[a].discard(b)
                if b in rem_adj:
                    rem_adj[b].discard(a)
                if a in rem_adj and not rem_adj[a]:
                    del rem_adj[a]
                if b in rem_adj and not rem_adj[b]:
                    del rem_adj[b]
    
    # If we didn't finish in time, fall back to single-edge paths for remaining
    if remaining:
        for e in remaining:
            a, b = tuple(e)
            paths.append([a, b])
    
    return paths

def _merge_paths(paths, edges, adj):
    """
    Try to merge pairs of paths that can be combined into a longer path.
    Two paths can be merged if one ends at the start of the other (or vice versa)
    and the combined path is simple and uses only graph edges.
    """
    edge_set = {frozenset(e) for e in edges}
    merged = True
    current_paths = list(paths)
    
    while merged:
        merged = False
        new_paths = []
        used = [False] * len(current_paths)
        
        for i in range(len(current_paths)):
            if used[i]:
                continue
            for j in range(i + 1, len(current_paths)):
                if used[j]:
                    continue
                
                p1 = current_paths[i]
                p2 = current_paths[j]
                
                # Try all four combinations: p1+p2, p1+reversed(p2), reversed(p1)+p2, reversed(p1)+reversed(p2)
                combined = None
                for rev1 in [False, True]:
                    a = list(reversed(p1)) if rev1 else list(p1)
                    for rev2 in [False, True]:
                        b = list(reversed(p2)) if rev2 else list(p2)
                        # Check if a's end connects to b's start
                        if a[-1] == b[0]:
                            candidate = a + b[1:]
                            # Check simplicity
                            if len(candidate) == len(set(candidate)):
                                # Check all edges are in graph
                                valid = True
                                for k in range(len(candidate) - 1):
                                    e = frozenset((candidate[k], candidate[k+1]))
                                    if e not in edge_set:
                                        valid = False
                                        break
                                if valid:
                                    combined = candidate
                                    break
                    if combined:
                        break
                if combined:
                    new_paths.append(combined)
                    used[i] = True
                    used[j] = True
                    merged = True
                    break
            if merged:
                break
        
        if merged:
            for k in range(len(current_paths)):
                if not used[k]:
                    new_paths.append(current_paths[k])
            current_paths = new_paths
    
    return current_paths

def _solve_graph(edges, time_budget=15.0):
    """
    Solve the path decomposition for a single graph.
    """
    if not edges:
        return []
    
    adj = _build_graph(edges)
    
    # Start with greedy longest-path decomposition
    paths = _greedy_path_cover(edges, adj, time_limit=min(5.0, time_budget))
    
    # Verify validity
    if not _is_valid_decomposition(paths, edges):
        # Fallback: single edges
        paths = [[list(e)] for e in edges]
    
    # Try to merge paths to reduce count
    remaining_time = time_budget - 5.0
    if remaining_time > 1.0:
        try:
            merged_paths = _merge_paths(paths, edges, adj)
            if _is_valid_decomposition(merged_paths, edges):
                paths = merged_paths
        except Exception:
            pass
    
    # Try iterative merging again
    if time.time() < _deadline - 10:
        try:
            merged_paths = _merge_paths(paths, edges, adj)
            if _is_valid_decomposition(merged_paths, edges):
                paths = merged_paths
        except Exception:
            pass
    
    return paths

def construct():
    start_time = time.time()
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    
    results = []
    for spec in specs:
        elapsed = time.time() - start_time
        if elapsed > _DEADLINE - 10:
            # Not enough time, use single-edge fallback for remaining
            edges = _instance(*spec)
            results.append([[list(e)] for e in edges])
            continue
        
        edges = _instance(*spec)
        budget = min(30.0, _DEADLINE - 10 - elapsed)
        paths = _solve_graph(edges, time_budget=budget)
        
        # Final validation
        if not _is_valid_decomposition(paths, edges):
            paths = [[list(e)] for e in edges]
        
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
