import time
import random
from collections import defaultdict

def construct():
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle(adj, edges_set, start_time_limit=0.1):
        """
        Finds a cycle in the graph defined by adj and edges_set.
        Returns a list of vertices representing the cycle, or None if no cycle exists.
        Tries to find a relatively long cycle.
        """
        if not edges_set:
            return None
        
        # Identify non-bridge edges to focus on
        # A simple way: try each edge
        # To be efficient, we can pick a random edge or one with high degree endpoints
        # Let's pick an edge (u, v)
        
        # Heuristic: Pick an edge (u, v) such that u and v are connected by a path in G - (u, v)
        # We can just iterate through edges
        
        best_cycle = None
        best_len = 0
        
        # To avoid iterating through ALL edges if the graph is dense, 
        # we can just pick a few candidates or use a heuristic.
        # For N=30, iterating through all edges is fine (max ~435 edges).
        
        candidates = list(edges_set)
        random.shuffle(candidates) # Randomize to avoid bias
        
        for u, v in candidates:
            # Remove edge (u, v) temporarily
            edge_to_remove = (min(u, v), max(u, v))
            if edge_to_remove not in edges_set:
                continue
            
            # We need to find a path from u to v in G - {(u, v)}
            # Use DFS
            visited = {u}
            path = [u]
            
            # DFS to find path from u to v
            # We want a long path, so we can try to extend as much as possible
            # But standard DFS finds *a* path.
            # To find a *longer* path, we might need backtracking.
            # Given time constraints, let's just find *a* path first.
            # If we find a path, we have a cycle.
            
            # Actually, finding the longest path is hard.
            # Let's just find *any* path.
            # We can use a BFS or DFS.
            
            def dfs(node, target, path, visited):
                if node == target:
                    return path[:]
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        # Check if edge (node, neighbor) is in edges_set
                        # And if it's not the removed edge
                        e1 = (min(node, neighbor), max(node, neighbor))
                        if e1 in edges_set and e1 != edge_to_remove:
                            visited.add(neighbor)
                            path.append(neighbor)
                            result = dfs(neighbor, target, path, visited)
                            if result:
                                return result
                            path.pop()
                            visited.remove(neighbor)
                return None

            # To encourage longer cycles, we can try to start DFS from a vertex with high degree
            # Or just run DFS.
            
            # Optimization: If the graph is small, we can try to find the longest path using a heuristic
            # But for now, let's just find a path.
            
            # We need to be careful with recursion depth and time.
            # N is small, so recursion depth is <= 30.
            
            # Let's run the DFS
            # We need to ensure we don't use the removed edge
            # The DFS above checks `e1 != edge_to_remove`
            
            # However, the DFS above is a standard DFS that returns the FIRST path found.
            # To get a longer path, we might need to explore more.
            # Let's try a different approach: 
            # Find the shortest path? No, we want long cycles.
            # Actually, any cycle is valid. The length determines the score.
            # A longer cycle is better.
            
            # Let's try to find a cycle that visits the most vertices.
            # We can use a heuristic: 
            # 1. Find a cycle.
            # 2. Try to extend it? No, edges are disjoint.
            
            # Let's stick to finding *a* cycle, but maybe prefer edges that are in dense parts.
            
            # Let's just find the first cycle we find.
            # To make it better, we can shuffle the neighbors in adj to randomize the path.
            
            path_found = dfs(u, v, [u], {u})
            if path_found:
                # The cycle is path_found + [u] (implicit close)
                # But path_found ends with v.
                # So the cycle vertices are path_found
                # The edges are (path[0], path[1]), ..., (path[-1], path[-2])? No.
                # Path is u -> ... -> v.
                # Cycle is u -> ... -> v -> u.
                # The vertices are path_found.
                # The edges are (path[i], path[i+1]) for i in 0..len-2, and (v, u).
                
                # Check length
                # If this cycle is longer than best_len, update best
                # But we are iterating over edges. We should pick the best one.
                
                # However, DFS is fast. Let's just take the first one found to save time?
                # No, let's collect a few and pick the longest.
                
                # Actually, the DFS finds a simple path.
                # The cycle length is len(path_found) + 1 (edges).
                # The number of vertices is len(path_found).
                
                # Let's just return the first cycle found to keep it simple and fast.
                # We can refine later if needed.
                return path_found
        
        return None

    def decompose(edges):
        n = 0
        for u, v in edges:
            n = max(n, u, v)
        n += 1
        
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        edges_set = set(tuple(sorted(e)) for e in edges)
        pieces = []
        
        while edges_set:
            cycle = find_cycle(adj, edges_set)
            if cycle:
                pieces.append(cycle)
                # Remove edges of the cycle
                for i in range(len(cycle) - 1):
                    u, v = cycle[i], cycle[i+1]
                    e = (min(u, v), max(u, v))
                    edges_set.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
                # Close the cycle
                u, v = cycle[-1], cycle[0]
                e = (min(u, v), max(u, v))
                if e in edges_set:
                    edges_set.remove(e)
                    adj[u].remove(v)
                    adj[v].remove(u)
            else:
                # No cycles left. All remaining edges are bridges.
                for e in edges_set:
                    pieces.append(list(e))
                break
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
        
    return results
