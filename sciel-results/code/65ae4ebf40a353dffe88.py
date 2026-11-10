import time
import random
from collections import defaultdict
from random import Random

def construct():
    start_time = time.time()
    TIME_LIMIT = 110.0  # Safety margin for 120s limit
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        # Path edges
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add(tuple(sorted((u, v))))
        # Random edges
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    def get_neighbors(node, adj):
        return list(adj[node])

    def try_find_cycle(u, v, remaining, adj, n):
        # Find a path from v to u in G - (u,v)
        # We want a simple path.
        path = [v]
        visited = {v}
        current = v
        
        while True:
            neighbors = get_neighbors(current, adj)
            random.shuffle(neighbors)
            
            moved = False
            for nxt in neighbors:
                # If nxt is u
                if nxt == u:
                    # We can close the cycle if the path has at least 2 edges
                    # Path is v ... current -> u
                    # Number of edges in path so far = len(path)
                    # Total edges in cycle = len(path) + 1
                    # We need total edges >= 3, so len(path) >= 2
                    if len(path) >= 2:
                        # Cycle found: path + [u]
                        # Vertices: path[0], ..., path[-1], u
                        cycle_vertices = path + [u]
                        # Construct edges
                        cycle_edges = []
                        for i in range(len(cycle_vertices) - 1):
                            a, b = cycle_vertices[i], cycle_vertices[i+1]
                            cycle_edges.append(tuple(sorted((a, b))))
                        return cycle_vertices, cycle_edges
                # If nxt is not visited and not u (handled above)
                elif nxt not in visited:
                    current = nxt
                    path.append(nxt)
                    visited.add(nxt)
                    moved = True
                    break
            
            if not moved:
                # Stuck at 'current'
                # Check if 'current' connects to any previous node in path
                # To maximize cycle length, connect to the earliest node
                best_idx = -1
                
                for i in range(len(path) - 1):
                    node = path[i]
                    edge = tuple(sorted((current, node)))
                    if edge in remaining:
                        best_idx = i
                        break # Earliest index gives longest cycle
                
                if best_idx != -1:
                    # Cycle is path[best_idx] ... current
                    cycle_vertices = path[best_idx:] + [path[best_idx]]
                    cycle_edges = []
                    for i in range(len(cycle_vertices) - 1):
                        a, b = cycle_vertices[i], cycle_vertices[i+1]
                        cycle_edges.append(tuple(sorted((a, b))))
                    return cycle_vertices, cycle_edges
                else:
                    return None, None

    def find_best_cycle(remaining, adj, n):
        if not remaining:
            return None, None
        
        candidates = list(remaining)
        random.shuffle(candidates)
        
        best_cycle_vertices = None
        best_cycle_edges = None
        max_len = 0
        
        # Try a few random edges to find a long cycle
        for u, v in candidates[:10]:
            vertices, edges = try_find_cycle(u, v, remaining, adj, n)
            if vertices:
                if len(vertices) > max_len:
                    max_len = len(vertices)
                    best_cycle_vertices = vertices
                    best_cycle_edges = edges
                    if max_len == n: # Hamiltonian cycle found
                        break
        
        return best_cycle_vertices, best_cycle_edges

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    
    for n, p, seed in specs:
        edges = instance(n, p, seed)
        remaining = set(edges)
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        pieces = []
        
        while remaining:
            if time.time() - start_time > TIME_LIMIT:
                # Time out: add all remaining edges as single pieces
                for e in remaining:
                    pieces.append(list(e))
                break
            
            vertices, cycle_edges = find_best_cycle(remaining, adj, n)
            
            if vertices and cycle_edges:
                # Add cycle as a piece
                pieces.append(vertices)
                
                # Remove edges from remaining and adj
                for e in cycle_edges:
                    remaining.remove(e)
                    u, v = e
                    adj[u].remove(v)
                    adj[v].remove(u)
            else:
                # No cycle found, take a single edge
                e = random.choice(list(remaining))
                pieces.append(list(e))
                remaining.remove(e)
                u, v = e
                adj[u].remove(v)
                adj[v].remove(u)
        
        results.append(pieces)
        
    return results
