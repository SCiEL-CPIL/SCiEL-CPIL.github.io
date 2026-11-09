"""Path decomposition via greedy long-path extraction."""
import time
import random as _random

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, deadline, max_iter=None):
    """Greedy path decomposition: repeatedly extract long simple paths."""
    if not edges:
        return []
    
    # Build adjacency: adj[u] = list of (v, edge_tuple)
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    
    # Track which edges are used: set of frozenset or sorted tuple
    used = set()
    for e in edges:
        used.add(tuple(sorted(e)))
    
    # remaining_edges count per vertex
    rem_deg = {v: len(adj[v]) for v in adj}
    
    paths = []
    all_edges_set = set(tuple(sorted(e)) for e in edges)
    total_edges = len(all_edges_set)
    
    # We'll work with a mutable structure: for each vertex, track available neighbors
    # via a dict of sets
    avail = {v: set() for v in adj}
    for u, v in edges:
        avail[u].add(v)
        avail[v].add(u)
    
    def extract_path(start):
        """Extract a simple path starting from 'start' using unused edges."""
        path = [start]
        visited = {start}
        cur = start
        while True:
            # Find an unused edge from cur to a neighbor
            # Prefer neighbors with higher remaining degree (to avoid dead ends)
            candidates = []
            for nb in avail[cur]:
                if nb not in visited:
                    # Check edge (cur, nb) is unused
                    e = tuple(sorted((cur, nb)))
                    if e not in used:
                        candidates.append(nb)
            if not candidates:
                break
            # Choose the candidate with highest remaining degree (greedy heuristic)
            # Ties broken by lower vertex id for determinism
            candidates.sort(key=lambda x: (-len(avail[x]), x))
            nxt = candidates[0]
            e = tuple(sorted((cur, nxt)))
            used.add(e)
            avail[cur].discard(nxt)
            avail[nxt].discard(cur)
            path.append(nxt)
            visited.add(nxt)
            cur = nxt
        return path
    
    # Strategy: repeatedly pick a start vertex with highest remaining degree,
    # extract a path, and continue until no edges remain.
    # To encourage long paths, start from vertices that are "central" in the remaining graph.
    
    iter_count = 0
    while used != all_edges_set:
        if time.time() > deadline:
            break
        if max_iter and iter_count >= max_iter:
            break
        iter_count += 1
        
        # Find vertices with remaining unused edges
        active_vertices = [v for v in avail if avail[v]]
        if not active_vertices:
            break
        
        # Choose start: vertex with max remaining degree
        # Among ties, pick the one that appears in more remaining edges (centrality)
        # Simple heuristic: max remaining degree
        start = max(active_vertices, key=lambda v: (len(avail[v]), -v))
        
        path = extract_path(start)
        if len(path) >= 2:  # at least one edge
            paths.append(path)
        else:
            # Single vertex, no edge extracted - shouldn't happen if active_vertices non-empty
            # Force pick an edge
            found = False
            for v in active_vertices:
                for nb in list(avail[v]):
                    e = tuple(sorted((v, nb)))
                    if e not in used:
                        path = [v, nb]
                        used.add(e)
                        avail[v].discard(nb)
                        avail[nb].discard(v)
                        paths.append(path)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
    
    return paths

def construct():
    deadline = time.time() + 120.0  # safety margin before 150s kill
    
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = _instance(*spec)
        paths = _decompose(edges, deadline)
        results.append(paths)
        # Check time
        if time.time() > deadline:
            # Fill remaining with trivial paths (each edge as path)
            # But we need to cover remaining edges
            # For safety, just use what we have + remaining as single-edge paths
            break
    
    # If we ran out of time, fill in remaining edges as single-edge paths
    # Actually, let's verify coverage and fix if needed
    for i, spec in enumerate(specs):
        edges = _instance(*spec)
        edge_set = set(tuple(sorted(e)) for e in edges)
        covered = set()
        for p in results[i]:
            for j in range(len(p) - 1):
                e = tuple(sorted((p[j], p[j+1])))
                covered.add(e)
        # Add missing edges as single-edge paths
        missing = edge_set - covered
        for e in missing:
            results[i].append(list(e))
    
    return results
