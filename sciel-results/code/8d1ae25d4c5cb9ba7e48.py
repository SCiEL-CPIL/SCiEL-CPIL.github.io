"""Decompose three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import random
from collections import defaultdict, deque

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def decompose(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Convert to set for fast lookup and removal
        edge_set = set(edges)
        
        def get_neighbors(v):
            """Get neighbors of v that are still connected by an edge."""
            return [u for u in adj[v] if (min(v,u), max(v,u)) in edge_set]
        
        def remove_edge(u, v):
            key = (min(u,v), max(v,u))
            edge_set.discard(key)
        
        def add_edge(u, v):
            key = (min(u,v), max(u,v))
            edge_set.add(key)
        
        pieces = []
        remaining_edges = set(edges)
        
        # Strategy: repeatedly find and remove cycles
        # Use a greedy approach: find any cycle in the remaining graph
        # Then remove its edges and add it as a piece
        # Continue until no cycles remain, then add remaining edges individually
        
        # We'll use a more sophisticated approach:
        # 1. Find all cycles using DFS-based cycle detection
        # 2. Prefer longer cycles (they cover more edges per piece)
        
        # For efficiency, we'll use a heuristic:
        # - Find a cycle using DFS from a random starting vertex
        # - Prefer to find longer cycles by exploring more before backtracking
        
        # Better approach: Use the fact that we want to maximize cycle length
        # We'll use a modified DFS that tries to find long cycles
        
        # Actually, let's use a simpler but effective approach:
        # Repeatedly find any cycle (using DFS), remove it, continue.
        # The key insight: longer cycles are better, so we should try to find them.
        
        # Let's use a different strategy: 
        # 1. While there exists a cycle in the remaining graph:
        #    a. Find a cycle (preferably a long one)
        #    b. Remove all edges in the cycle
        #    c. Add the cycle as a piece
        # 2. Add all remaining edges as individual pieces
        
        # To find cycles efficiently, we can use the following:
        # - For each vertex, do a DFS to find a back edge, which indicates a cycle
        # - The cycle is the path from the current vertex to the ancestor + the back edge
        
        # Let's implement this:
        
        def find_cycle(adj_remaining):
            """Find a cycle in the graph represented by adj_remaining.
            Returns the cycle as a list of vertices, or None if no cycle exists."""
            visited = set()
            parent = {}
            cycle = None
            
            def dfs(u, start):
                nonlocal cycle
                visited.add(u)
                for v in adj_remaining.get(u, []):
                    if v not in visited:
                        parent[v] = u
                        dfs(v, start)
                        if cycle:
                            return
                    elif v != parent.get(u, -1) and v in visited:
                        # Found a back edge to an ancestor
                        # Reconstruct the cycle
                        cycle_nodes = [u]
                        curr = u
                        while curr != v:
                            curr = parent[curr]
                            cycle_nodes.append(curr)
                        cycle_nodes.reverse()
                        cycle = cycle_nodes
                        return
                
            for u in list(adj_remaining.keys()):
                if u not in visited:
                    dfs(u, u)
                    if cycle:
                        return cycle
            return None
        
        # However, DFS-based cycle finding might find short cycles.
        # Let's use a better approach: find the longest cycle possible.
        # This is NP-hard in general, but for small graphs we can use heuristics.
        
        # Alternative: use a randomized approach with local search
        # Or: use the fact that we can greedily find cycles and it's good enough
        
        # Let's try a different approach: 
        # Use a maximum cycle packing heuristic
        # 1. While there's a cycle:
        #    a. Find a cycle using BFS/DFS
        #    b. Remove it
        # 2. Add remaining edges
        
        # To find longer cycles, we can use a modified DFS that explores deeper first
        
        def find_long_cycle(adj_remaining):
            """Try to find a relatively long cycle in the graph."""
            # Use DFS with a bias towards exploring unvisited nodes first
            # and trying to close cycles only when necessary
            
            best_cycle = None
            
            def dfs(u, path, visited_in_path):
                nonlocal best_cycle
                for v in adj_remaining.get(u, []):
                    if v == path[0] and len(path) >= 3:
                        # Found a cycle back to the start
                        cycle = path[:]
                        if best_cycle is None or len(cycle) > len(best_cycle):
                            best_cycle = cycle
                    elif v not in visited_in_path:
                        visited_in_path.add(v)
                        dfs(v, path + [v], visited_in_path)
                        visited_in_path.remove(v)
                        if best_cycle and len(best_cycle) >= len(path) + 1:
                            # Prune: no need to explore further if we already have a good cycle
                            pass
            
            # Try from each vertex
            for start in adj_remaining:
                if len(adj_remaining[start]) == 0:
                    continue
                dfs(start, [start], {start})
                if best_cycle and len(best_cycle) >= 10:
                    break  # Good enough
            
            return best_cycle
        
        # This DFS approach might be too slow for larger graphs.
        # Let's use a simpler but faster approach:
        
        # Use a greedy cycle-finding algorithm:
        # 1. Pick a vertex with high degree
        # 2. Do a DFS to find a cycle
        # 3. Remove the cycle
        # 4. Repeat
        
        # For the given graph sizes (20-30 vertices), even a simple approach should work
        
        def find_cycle_simple(adj_remaining):
            """Find any cycle in the graph using DFS."""
            visited = set()
            parent = {}
            
            def dfs(u):
                visited.add(u)
                for v in adj_remaining.get(u, []):
                    if v not in visited:
                        parent[v] = u
                        result = dfs(v)
                        if result:
                            return result
                    elif v != parent.get(u, -1):
                        # Found a cycle
                        cycle = [u]
                        curr = u
                        while curr != v:
                            curr = parent[curr]
                            cycle.append(curr)
                        cycle.reverse()
                        return cycle
                return None
            
            for u in adj_remaining:
                if u not in visited:
                    result = dfs(u)
                    if result:
                        return result
            return None
        
        # Let's use a hybrid approach:
        # 1. Try to find a long cycle using a limited DFS
        # 2. If that fails or finds a short cycle, use simple cycle finding
        
        while True:
            # Build adjacency from remaining edges
            adj_remaining = defaultdict(list)
            for u, v in remaining_edges:
                adj_remaining[u].append(v)
                adj_remaining[v].append(u)
            
            # Remove isolated vertices
            adj_remaining = {k: v for k, v in adj_remaining.items() if len(v) > 0}
            
            if not adj_remaining:
                break
            
            # Try to find a cycle
            cycle = find_cycle_simple(adj_remaining)
            
            if cycle is None:
                break
            
            # Remove edges in the cycle
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                remaining_edges.discard((min(u,v), max(u,v)))
            
            pieces.append(cycle)
        
        # Add remaining edges as individual pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
