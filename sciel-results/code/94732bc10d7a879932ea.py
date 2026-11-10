"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random
from collections import defaultdict, deque

def _instance(n, p, seed):
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_graph(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _edge_path_cover_greedy(edges, adj):
    """
    Greedy edge path cover: repeatedly find the longest simple path
    in the remaining edge set, remove its edges, repeat.
    """
    remaining_edges = set(edges)
    paths = []
    
    while remaining_edges:
        # Find a starting vertex with highest degree in remaining graph
        deg = defaultdict(int)
        for u, v in remaining_edges:
            deg[u] += 1
            deg[v] += 1
        
        # Pick start vertex with max degree
        if not deg:
            break
        start = max(deg.keys(), key=lambda x: deg[x])
        
        # Greedy walk: at each step, extend to a neighbor with max degree
        # among unvisited neighbors (to keep path simple)
        visited = {start}
        path = [start]
        current = start
        
        while True:
            # Find neighbors of current in remaining edges
            candidates = []
            for u, v in remaining_edges:
                if u == current and v not in visited:
                    candidates.append(v)
                elif v == current and u not in visited:
                    candidates.append(u)
            
            if not candidates:
                break
            
            # Pick neighbor with highest remaining degree
            best_next = max(candidates, key=lambda x: deg.get(x, 0))
            path.append(best_next)
            visited.add(best_next)
            
            # Remove edges from current to best_next
            e1 = (current, best_next) if current < best_next else (best_next, current)
            if e1 in remaining_edges:
                remaining_edges.remove(e1)
            
            current = best_next
        
        paths.append(path)
    
    return paths

def _path_cover_with_restarts(edges, adj, n_restarts=50):
    """
    Try multiple random restarts with different start vertices and
    greedy heuristics to find a better (fewer paths) cover.
    """
    best_paths = None
    best_count = float('inf')
    
    rng = Random(42)
    
    for restart in range(n_restarts):
        remaining_edges = set(edges)
        paths = []
        
        while remaining_edges:
            deg = defaultdict(int)
            for u, v in remaining_edges:
                deg[u] += 1
                deg[v] += 1
            
            if not deg:
                break
            
            # Randomly pick a start vertex, weighted by degree
            vertices = list(deg.keys())
            weights = [deg[v] for v in vertices]
            start = rng.choices(vertices, weights=weights, k=1)[0]
            
            visited = {start}
            path = [start]
            current = start
            
            while True:
                candidates = []
                for u, v in remaining_edges:
                    if u == current and v not in visited:
                        candidates.append(v)
                    elif v == current and u not in visited:
                        candidates.append(u)
                
                if not candidates:
                    break
                
                # Weighted random choice by degree
                weights = [deg.get(x, 0) for x in candidates]
                best_next = rng.choices(candidates, weights=weights, k=1)[0]
                path.append(best_next)
                visited.add(best_next)
                
                e1 = (current, best_next) if current < best_next else (best_next, current)
                if e1 in remaining_edges:
                    remaining_edges.remove(e1)
                
                current = best_next
            
            paths.append(path)
        
        if len(paths) < best_count:
            best_count = len(paths)
            best_paths = paths
        
        # Early stop if we hit the theoretical lower bound
        # Lower bound: max(1, ceil(max_degree / 2)) for each connected component, but
        # a simpler bound: number of odd-degree vertices / 2
        odd_deg = sum(1 for v in deg if deg[v] % 2 == 1)
        lower_bound = max(1, odd_deg // 2)
        if best_count <= lower_bound:
            break
    
    return best_paths if best_paths is not None else _edge_path_cover_greedy(edges, adj)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    results = []
    for n, p, seed in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = _instance(n, p, seed)
        adj = _build_graph(edges)
        
        # Use restart-based greedy to minimize path count
        paths = _path_cover_with_restarts(edges, adj, n_restarts=100)
        results.append(paths)
        
        # Check deadline
        if time.time() > deadline:
            # Fall back to simple greedy for remaining graphs
            for n2, p2, seed2 in ((25, 0.3, 2), (30, 0.25, 3)):
                if len(results) >= 3:
                    break
                edges2 = _instance(n2, p2, seed2)
                adj2 = _build_graph(edges2)
                paths2 = _edge_path_cover_greedy(edges2, adj2)
                results.append(paths2)
            break
    
    return results
# EVOLVE-BLOCK-END
