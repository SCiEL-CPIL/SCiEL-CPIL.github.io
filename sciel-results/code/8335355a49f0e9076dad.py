import time
import sys
from random import Random

# Increase recursion limit for DFS on potentially deep paths, though N is small
sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []

    for spec in specs:
        # Check if we have enough time to process this graph
        # If not, we still need to return a valid result.
        # We will process it but with a tighter deadline or just trivial if time is very low.
        
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            
        remaining_edges = set(edges)
        pieces = []
        
        while remaining_edges:
            if time.time() > deadline:
                # Emergency exit: add all remaining edges as singles
                for e in remaining_edges:
                    pieces.append(list(e))
                break
            
            # Find the longest cycle in the current graph
            best_cycle = None
            best_len = 0
            
            # Define DFS
            def dfs_cycle(start, path, visited):
                nonlocal best_cycle, best_len
                current = path[-1]
                
                # Pruning: even if we visit all remaining nodes, can we beat best_len?
                if len(path) + (n - len(visited)) <= best_len:
                    return
                
                # Sort neighbors by degree descending to find long cycles faster
                # This creates a list, which is fine for small N
                neighbors = sorted(adj[current], key=lambda x: -len(adj[x]))
                
                for neighbor in neighbors:
                    if neighbor == start and len(path) >= 3:
                        # Found a cycle
                        if len(path) > best_len:
                            best_len = len(path)
                            best_cycle = list(path)
                        # We can continue searching for longer cycles
                    elif neighbor not in visited:
                        visited.add(neighbor)
                        path.append(neighbor)
                        dfs_cycle(start, path, visited)
                        path.pop()
                        visited.remove(neighbor)
                        # If we found a Hamiltonian cycle, we can stop
                        if best_len == n:
                            return

            # Iterate over all vertices with degree >= 2
            # We can optimize by only trying vertices that are part of potential cycles
            # but for N=30, trying all is cheap enough if DFS prunes well.
            for v in range(n):
                if len(adj[v]) >= 2:
                    dfs_cycle(v, [v], {v})
                    if best_len == n:
                        break
            
            if best_cycle:
                # Extract edges from the cycle
                cycle_edges = set()
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = tuple(sorted((u, v)))
                    cycle_edges.add(edge)
                
                # Remove these edges from remaining and adj
                for e in cycle_edges:
                    if e in remaining_edges:
                        remaining_edges.remove(e)
                    u, v = e
                    if v in adj[u]:
                        adj[u].remove(v)
                    if u in adj[v]:
                        adj[v].remove(u)
                
                pieces.append(best_cycle)
            else:
                # No cycles found, remaining edges are bridges or isolated
                for e in remaining_edges:
                    pieces.append(list(e))
                break
        
        results.append(pieces)

    return results
