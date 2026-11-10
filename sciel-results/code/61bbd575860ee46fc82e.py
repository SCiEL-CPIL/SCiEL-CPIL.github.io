"""Edge path cover via greedy longest-path extraction with structural heuristics."""
import time
import sys
from random import Random

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adjacency(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    for u in adj:
        adj[u] = list(adj[u])
    return adj


def _greedy_path_cover(edges, adj, deadline_frac=0.85, start_time=None):
    """
    Greedy: repeatedly find a long simple path using unused edges, remove its edges, repeat.
    Uses a heuristic walk: at each step prefer the neighbor that leads to the longest
    continuation (look-ahead depth limited for speed).
    """
    if start_time is None:
        start_time = time.time()
    
    remaining_edges = set(edges)
    # Track remaining degree for each vertex
    rem_deg = {}
    for u, v in remaining_edges:
        rem_deg[u] = rem_deg.get(u, 0) + 1
        rem_deg[v] = rem_deg.get(v, 0) + 1
    
    # Build adjacency from remaining edges (mutable)
    rem_adj = {u: set() for u in rem_deg}
    for u, v in remaining_edges:
        rem_adj[u].add(v)
        rem_adj[v].add(u)
    
    paths = []
    
    while remaining_edges:
        if (time.time() - start_time) / 120.0 > deadline_frac:
            break
        
        # Choose start vertex: prefer odd-degree, then highest degree
        odd_verts = [u for u in rem_adj if rem_deg.get(u, 0) % 2 == 1]
        if odd_verts:
            start = max(odd_verts, key=lambda u: rem_deg.get(u, 0))
        else:
            start = max(rem_adj.keys(), key=lambda u: rem_deg.get(u, 0))
        
        # Walk: greedily extend path
        path = [start]
        used_in_path = set()
        prev = None
        cur = start
        
        while True:
            # Find next vertex: prefer one that doesn't create a dead end
            candidates = [v for v in rem_adj[cur] if v not in used_in_path and v != prev]
            if not candidates:
                break
            
            # Heuristic: pick candidate with highest remaining degree (to avoid trapping)
            # But also avoid going to a vertex that would isolate others
            best = None
            best_score = -1
            for v in candidates:
                # Score: prefer higher degree, penalize if v has degree 1 (would be endpoint)
                deg = rem_deg.get(v, 0)
                # Avoid going to degree-1 vertex unless it's the only option
                score = deg * 10
                if deg == 1 and len(candidates) > 1:
                    score -= 100
                # Slight preference for vertices with more unvisited neighbors
                unvisited_neigh = len([x for x in rem_adj[v] if x not in used_in_path and x != cur])
                score += unvisited_neigh
                if score > best_score:
                    best_score = score
                    best = v
            
            if best is None:
                break
            
            # Remove edges along the step
            edge = tuple(sorted((cur, best)))
            if edge not in remaining_edges:
                break
            remaining_edges.discard(edge)
            rem_adj[cur].discard(best)
            rem_adj[best].discard(cur)
            rem_deg[cur] -= 1
            rem_deg[best] -= 1
            
            used_in_path.add(cur)
            prev = cur
            cur = best
            path.append(cur)
        
        if len(path) > 1:
            paths.append(path)
        else:
            # Single vertex, shouldn't happen if edges remain, but handle it
            # Find any remaining edge incident to start
            if rem_adj.get(start):
                v = next(iter(rem_adj[start]))
                edge = tuple(sorted((start, v)))
                if edge in remaining_edges:
                    remaining_edges.discard(edge)
                    rem_adj[start].discard(v)
                    rem_adj[v].discard(start)
                    rem_deg[start] -= 1
                    rem_deg[v] -= 1
                    paths.append([start, v])
    
    return paths


def _try_improve(paths, edges, adj, deadline_frac=0.9, start_time=None):
    """
    Try to reduce path count by merging adjacent paths where possible.
    Two paths can be merged if the end of one is adjacent to the start of another
    via an unused edge... but all edges are used. Instead, we can try to re-route.
    For now, this is a placeholder for future improvement.
    """
    return paths


def construct():
    start_time = time.time()
    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for spec in specs:
        edges = _instance(*spec)
        adj = _build_adjacency(edges)
        
        # Try multiple strategies and pick the best
        best_paths = None
        best_count = float('inf')
        
        # Strategy 1: Greedy with standard heuristic
        paths1 = _greedy_path_cover(edges, adj, deadline_frac=0.7, start_time=start_time)
        if len(paths1) < best_count:
            best_count = len(paths1)
            best_paths = paths1
        
        if (time.time() - start_time) / 120.0 > 0.8:
            results.append(best_paths if best_paths else [[list(e) for e in edges]])
            continue
        
        # Strategy 2: Greedy with different start selection (lowest degree first)
        # We can modify by trying a different approach: start from leaves
        edges2 = list(edges)
        paths2 = []
        rem_edges2 = set(edges)
        rem_deg2 = {}
        for u, v in rem_edges2:
            rem_deg2[u] = rem_deg2.get(u, 0) + 1
            rem_deg2[v] = rem_deg2.get(v, 0) + 1
        rem_adj2 = {u: set() for u in rem_deg2}
        for u, v in rem_edges2:
            rem_adj2[u].add(v)
            rem_adj2[v].add(u)
        
        while rem_edges2:
            if (time.time() - start_time) / 120.0 > 0.85:
                break
            # Start from leaf (degree 1) if exists, else lowest degree
            leaves = [u for u in rem_adj2 if rem_deg2.get(u, 0) == 1]
            if leaves:
                start = leaves[0]
            else:
                start = min(rem_adj2.keys(), key=lambda u: rem_deg2.get(u, 0))
            
            path = [start]
            used_in_path = set()
            prev = None
            cur = start
            
            while True:
                candidates = [v for v in rem_adj2[cur] if v not in used_in_path and v != prev]
                if not candidates:
                    break
                # Pick highest degree neighbor
                best = max(candidates, key=lambda v: rem_deg2.get(v, 0))
                
                edge = tuple(sorted((cur, best)))
                if edge not in rem_edges2:
                    break
                rem_edges2.discard(edge)
                rem_adj2[cur].discard(best)
                rem_adj2[best].discard(cur)
                rem_deg2[cur] -= 1
                rem_deg2[best] -= 1
                
                used_in_path.add(cur)
                prev = cur
                cur = best
                path.append(cur)
            
            if len(path) > 1:
                paths2.append(path)
        
        if len(paths2) < best_count:
            best_count = len(paths2)
            best_paths = paths2
        
        results.append(best_paths if best_paths else [[list(e) for e in edges]])
    
    return results
