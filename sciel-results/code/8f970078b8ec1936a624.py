"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def _instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _build_adj(edges):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _find_cycle(adj, remaining_edges, start, deadline):
    """
    Find a simple cycle starting and ending at 'start' using DFS.
    Returns a list of vertices [v0, v1, ..., vk] where v0=start, vk is neighbor of start.
    Implies edges (v0,v1)...(vk-1,vk)(vk,v0).
    """
    path = [start]
    visited = {start}
    
    # Sort neighbors by degree to find longer cycles (heuristic)
    def get_sorted_neighbors(node):
        return sorted(adj[node], key=lambda x: len(adj[x]), reverse=True)

    def dfs(node):
        if time.time() > deadline:
            return None
            
        # Check if we can close the cycle
        # We are at 'node'. If 'node' is connected to 'start' and path length >= 2, we have a cycle.
        # But we must ensure the edge (node, start) exists in remaining_edges.
        if len(path) >= 2:
            e_close = tuple(sorted((node, start)))
            if e_close in remaining_edges:
                # Return the path (vertices)
                return path[:]
        
        neighbors = get_sorted_neighbors(node)
        for nb in neighbors:
            e = tuple(sorted((node, nb)))
            if e not in remaining_edges:
                continue
            
            if nb == start:
                # Only allow returning to start if we have enough vertices to form a cycle (>=3 vertices total)
                # path currently has 'node' at the end.
                # If we go to start, the cycle is path + [start].
                # Vertices in cycle: path[0]...path[-1], start.
                # Since path[0] is start, the distinct vertices are path.
                # We need len(path) >= 2 to have at least 2 edges in the path part, plus closing edge = 3 edges.
                # So len(path) >= 2 is correct for a cycle of length >= 3.
                if len(path) >= 2:
                    return path[:]
                else:
                    continue
            
            if nb in visited:
                continue
            
            visited.add(nb)
            path.append(nb)
            
            result = dfs(nb)
            if result:
                return result
            
            path.pop()
            visited.discard(nb)
            
        return None
    
    return dfs(start)

def _decompose_graph(edges, deadline):
    """
    Decompose graph into cycles and single edges.
    Greedy approach: Repeatedly find a cycle, remove its edges.
    """
    if not edges:
        return []
    
    remaining_edges = set(tuple(sorted(e)) for e in edges)
    adj = _build_adj(remaining_edges)
    pieces = []
    
    while remaining_edges:
        if time.time() > deadline:
            break
            
        # Find a vertex with degree >= 2 to start a cycle
        candidates = [v for v in adj if len(adj[v]) >= 2]
        if not candidates:
            # No more cycles possible. All remaining edges are bridges/leaves.
            # Add them as single edges.
            for e in remaining_edges:
                pieces.append([e[0], e[1]])
            remaining_edges.clear()
            break
        
        # Pick a start vertex. Heuristic: highest degree.
        start = max(candidates, key=lambda v: len(adj[v]))
        
        cycle = _find_cycle(adj, remaining_edges, start, deadline)
        
        if cycle and len(cycle) >= 2: # cycle list has vertices. len >= 2 means at least 2 vertices in path, so cycle length >= 3.
            # The cycle vertices are in 'cycle'.
            # Edges are (cycle[i], cycle[i+1]) for i in range(len(cycle)-1) and (cycle[-1], cycle[0]).
            pieces.append(cycle)
            
            # Remove edges from remaining
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                e = tuple(sorted((u, v)))
                remaining_edges.discard(e)
                if e in adj[u]: adj[u].discard(v)
                if e in adj[v]: adj[v].discard(u)
        else:
            # No cycle found from this start. 
            # This can happen if the component is a tree (no cycles).
            # We should extract single edges from this component or just add all remaining as single edges if we can't find cycles anywhere.
            # To be safe and simple, if we fail to find a cycle from the best candidate, 
            # we might be stuck in a forest. 
            # Let's just add all remaining edges as single edges to ensure termination.
            # This is suboptimal but valid.
            for e in remaining_edges:
                pieces.append([e[0], e[1]])
            remaining_edges.clear()
            break

    return pieces

def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    start_time = time.time()
    total_deadline = start_time + 140.0
    
    for i, spec in enumerate(specs):
        elapsed = time.time() - start_time
        remaining_time = total_deadline - elapsed
        per_graph_time = remaining_time / (len(specs) - i)
        graph_deadline = time.time() + per_graph_time * 0.95
        
        edges = _instance(*spec)
        pieces = _decompose_graph(edges, graph_deadline)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
