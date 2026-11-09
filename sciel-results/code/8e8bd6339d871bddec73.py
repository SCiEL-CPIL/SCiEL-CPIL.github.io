"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # 10s safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, time_limit=45.0):
        """Decompose edges into cycles and single edges to minimize piece count."""
        # Build adjacency
        adj = defaultdict(list)
        edge_set = set()
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            edge_set.add((u, v) if u < v else (v, u))
        
        # We'll greedily find cycles, preferring longer ones
        # Strategy: repeatedly find a cycle in the remaining graph,
        # remove its edges, continue until no cycle exists.
        # To get longer cycles, we can use a heuristic: find a cycle
        # through a specific vertex or use a longest-path-like approach.
        
        remaining_edges = set(edge_set)
        pieces = []
        
        # Build adjacency from remaining edges
        def build_adj(edge_set):
            adj = defaultdict(list)
            for u, v in edge_set:
                adj[u].append(v)
                adj[v].append(u)
            return adj
        
        # Find a cycle in the graph using DFS with a twist to prefer longer cycles
        def find_cycle(adj, edge_set):
            """Find a cycle, preferring longer ones."""
            if not edge_set:
                return None
            
            # Pick a starting vertex with highest degree
            deg = {v: len(neigh) for v, neigh in adj.items()}
            if not deg:
                return None
            start = max(deg, key=lambda v: deg[v])
            
            # Try to find a long cycle starting from 'start'
            # Use DFS with a visited set, trying to extend paths
            best_cycle = None
            best_len = 2  # minimum is 3 vertices (cycle of length 3)
            
            # We'll do a limited DFS to find cycles through 'start'
            # Path: start -> ... -> start
            visited = {start}
            path = [start]
            
            def dfs(current, path, visited):
                nonlocal best_cycle, best_len
                for nb in adj[current]:
                    edge = (min(current, nb), max(current, nb))
                    if edge not in edge_set:
                        continue
                    if nb == start and len(path) >= 3:
                        # Found a cycle
                        cycle = path[:]
                        if len(cycle) > best_len:
                            best_len = len(cycle)
                            best_cycle = cycle[:]
                        return
                    if nb not in visited:
                        visited.add(nb)
                        path.append(nb)
                        dfs(nb, path, visited)
                        path.pop()
                        visited.discard(nb)
            
            dfs(start, path, visited)
            
            if best_cycle:
                return best_cycle
            
            # If no cycle found through start with that method, try a simpler approach
            # Just find any cycle using standard DFS
            for u in sorted(adj.keys()):
                visited = set()
                stack = [(u, [u])]
                while stack:
                    node, path = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    for nb in adj[node]:
                        edge = (min(node, nb), max(node, nb))
                        if edge not in edge_set:
                            continue
                        if nb in path:
                            # Found cycle
                            idx = path.index(nb)
                            cycle = path[idx:]
                            if len(cycle) >= 3:
                                return cycle
                        elif nb not in visited:
                            stack.append((nb, path + [nb]))
            
            return None
        
        # Main loop: find cycles and remove their edges
        while time.time() - start_time < DEADLINE:
            adj = build_adj(remaining_edges)
            
            # Check if any vertex has degree >= 2 (necessary for a cycle)
            has_cycle = any(len(neigh) >= 2 for neigh in adj.values())
            if not has_cycle:
                break
            
            cycle = find_cycle(adj, remaining_edges)
            if not cycle:
                break
            
            # Remove cycle edges
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                edge = (min(u, v), max(u, v))
                if edge in remaining_edges:
                    remaining_edges.discard(edge)
            
            # Add cycle as a piece (list of vertices in order)
            pieces.append(cycle)
        
        # Remaining edges are single-edge pieces
        for u, v in remaining_edges:
            pieces.append([u, v])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        n, p, seed = spec
        edges = instance(n, p, seed)
        
        if time.time() - start_time > DEADLINE:
            # Fallback: all edges as single pieces
            results.append([list(e) for e in edges])
            continue
        
        pieces = decompose_graph(edges, n, time_limit=DEADLINE)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
