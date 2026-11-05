import time
import random
import sys
from collections import deque

sys.setrecursionlimit(100000)

def _instance(n, p, seed):
    """Copied from the verifier (must match exactly)."""
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _find_shortest_cycle(adj, n):
    """
    Find the shortest cycle in the graph.
    Returns list of vertices or None.
    """
    best_cycle = None
    best_len = float('inf')
    
    for start in range(n):
        if not adj[start]:
            continue
        
        # BFS from start
        dist = [-1] * n
        parent = [-1] * n
        dist[start] = 0
        q = deque([start])
        
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
                elif v != parent[u] and dist[v] < dist[u]: 
                    # Found a back edge to an ancestor (not parent)
                    # Cycle length: dist[u] - dist[v] + 1
                    cycle_len = dist[u] - dist[v] + 1
                    if cycle_len < best_len:
                        # Reconstruct cycle
                        cycle = [u]
                        curr = u
                        while curr != v:
                            curr = parent[curr]
                            cycle.append(curr)
                        # cycle is v...u, need to reverse to start from v?
                        # Actually, the cycle is v -> ... -> u -> v
                        # We have u -> ... -> v in reverse parent chain
                        # Let's just return the list, order doesn't matter for validity
                        # but let's make it consistent
                        cycle.reverse()
                        best_len = cycle_len
                        best_cycle = cycle
        
    return best_cycle

def _decompose_forest(adj, n):
    """
    Decompose the remaining forest into paths.
    adj is the adjacency list of the forest.
    Returns list of paths (each path is a list of vertices).
    """
    paths = []
    # Copy adj to mutable structure
    # We need to remove edges as we form paths
    # Using a set of edges for remaining edges
    remaining_edges = set()
    for u in range(n):
        for v in adj[u]:
            if u < v:
                remaining_edges.add((u, v))
                
    # Helper to get neighbors in remaining graph
    def get_neighbors(u):
        return [v for (a, b) in remaining_edges if a == u or b == u]
        
    while remaining_edges:
        # Find an odd-degree node
        start = -1
        for u in range(n):
            if len(get_neighbors(u)) % 2 == 1:
                start = u
                break
        if start == -1:
            # Should not happen for a forest with edges, but if it does, pick any
            for u in range(n):
                if get_neighbors(u):
                    start = u
                    break
            if start == -1: break
            
        # Walk from start to next odd node
        path = [start]
        current = start
        visited_edges = set()
        
        while True:
            neighbors = get_neighbors(current)
            # Filter out visited edges
            next_node = -1
            for v in neighbors:
                edge = (min(current, v), max(current, v))
                if edge not in visited_edges:
                    next_node = v
                    visited_edges.add(edge)
                    break
            
            if next_node == -1:
                break # Stuck? Shouldn't happen if we start at odd and end at odd
            
            # Check if next_node is odd (in the current remaining graph context?
            # No, degree changes as we remove edges.
            # We stop when we reach a node that has NO unvisited edges?
            # Or when we reach an odd node in the ORIGINAL forest?
            # The greedy algorithm: Walk until you hit a leaf or a node with no unvisited edges.
            # In a tree, this will end at a leaf (which is odd).
            
            path.append(next_node)
            current = next_node
            
            # Check if current has no more unvisited edges
            # If so, we are at the end of the path
            if not [v for v in get_neighbors(current) if (min(current, v), max(current, v)) not in visited_edges]:
                break
        
        # Remove the edges in this path from remaining_edges
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge = (min(u, v), max(u, v))
            remaining_edges.discard(edge)
            
        paths.append(path)
        
    return paths

def _solve_one(edges, n, deadline):
    # Build initial adjacency
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    cycles = []
    
    # Phase 1: Extract cycles
    while True:
        if time.time() > deadline:
            break
            
        cycle = _find_shortest_cycle(adj, n)
        if not cycle:
            break
            
        cycles.append(cycle)
        
        # Remove cycle edges
        m = len(cycle)
        for i in range(m):
            u = cycle[i]
            v = cycle[(i + 1) % m]
            adj[u].discard(v)
            adj[v].discard(u)
            
    # Phase 2: Decompose remaining forest
    paths = _decompose_forest(adj, n)
    
    return cycles + paths

def construct():
    start_time = time.time()
    total_budget = 140.0
    per_graph_budget = total_budget / 3.0
    
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = _instance(n, p, seed)
        graph_deadline = start_time + per_graph_budget * (i + 1)
        
        if graph_deadline > start_time + total_budget - 2.0:
            graph_deadline = start_time + total_budget - 2.0
            
        pieces = _solve_one(edges, n, graph_deadline)
        results.append(pieces)
        
        if time.time() > start_time + total_budget - 5.0:
            for j in range(i + 1, len(specs)):
                nn, pp, ss = specs[j]
                ee = _instance(nn, pp, ss)
                results.append([list(e) for e in ee])
            break
            
    return results
