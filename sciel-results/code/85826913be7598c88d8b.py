"""Decompositions of three fixed graphs into few cycles and edges (Erdős-Gallai)."""
# EVOLVE-BLOCK-START
import sys
from collections import defaultdict, deque
import heapq
from random import Random

def construct():
    # Generate the three instance graphs
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
        edges = instance(*spec)
        n = spec[0]
        result = decompose_graph(n, edges)
        results.append(result)
    
    return results

def decompose_graph(n, edges):
    """
    Decompose the undirected graph into minimum number of cycles and edges.
    Uses a greedy strategy to find long cycles by preferring high-degree vertices.
    """
    if not edges:
        return []
    
    # Build adjacency structure
    edge_set = set()
    for u, v in edges:
        edge_set.add((min(u,v), max(u,v)))
    
    pieces = []
    remaining_edges = set(edge_set)
    
    def get_adj(rem_edges):
        adj = defaultdict(set)
        for u, v in rem_edges:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def find_cycle_greedy(adj, n, rem_edges):
        """
        Find a cycle using a greedy strategy:
        Start from a high-degree node, move to the neighbor with the highest degree.
        If we hit a node in the current path, we have a cycle.
        """
        # Sort nodes by degree descending
        nodes_by_degree = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        
        for start in nodes_by_degree:
            if not adj[start]:
                continue
            
            # Try to extend a path from start
            path = [start]
            path_set = {start}
            curr = start
            
            # We will walk greedily
            while True:
                # Find best neighbor
                best_neighbor = None
                best_degree = -1
                
                for neighbor in adj[curr]:
                    # Edge (curr, neighbor) must be in rem_edges
                    e = (min(curr, neighbor), max(curr, neighbor))
                    if e not in rem_edges:
                        continue
                    
                    if neighbor in path_set:
                        # Found a cycle
                        idx = path.index(neighbor)
                        cycle = path[idx:]
                        if len(cycle) >= 3:
                            return cycle
                        else:
                            break
                    
                    # Heuristic: choose neighbor with highest degree in remaining graph
                    deg = len(adj[neighbor])
                    if deg > best_degree:
                        best_degree = deg
                        best_neighbor = neighbor
                
                if best_neighbor is None:
                    # Dead end, no cycle found from this start
                    break
                
                # Move to best_neighbor
                curr = best_neighbor
                path.append(curr)
                path_set.add(curr)
                
                # Safety break: if path gets too long, stop
                if len(path) > n:
                    break
            
        return None

    def find_cycle_dfs(adj, n, rem_edges):
        """Fallback: Standard DFS to find any cycle."""
        visited = [False] * n
        parent = [-1] * n
        
        def dfs(start):
            stack = [(start, iter(adj[start]))]
            visited[start] = True
            while stack:
                node, it = stack[-1]
                advanced = False
                for neighbor in it:
                    e = (min(node, neighbor), max(node, neighbor))
                    if e not in rem_edges:
                        continue
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        stack.append((neighbor, iter(adj[neighbor])))
                        advanced = True
                        break
                    elif neighbor != parent[node]:
                        # Found back edge
                        # Reconstruct cycle
                        cycle = [node]
                        curr = node
                        while curr != neighbor:
                            curr = parent[curr]
                            cycle.append(curr)
                        if len(cycle) >= 3:
                            # Verify
                            for i in range(len(cycle)):
                                u = cycle[i]
                                v = cycle[(i+1) % len(cycle)]
                                if (min(u,v), max(u,v)) not in rem_edges:
                                    return None
                            return cycle
                if not advanced:
                    stack.pop()
            return None
        
        for start in range(n):
            if not visited[start] and adj[start]:
                cycle = dfs(start)
                if cycle:
                    return cycle
        return None

    max_iterations = 1000
    iteration = 0
    while remaining_edges and iteration < max_iterations:
        iteration += 1
        rem_adj = get_adj(remaining_edges)
        
        # Try greedy first
        cycle = find_cycle_greedy(rem_adj, n, remaining_edges)
        
        if cycle is None:
            # Fallback to DFS
            cycle = find_cycle_dfs(rem_adj, n, remaining_edges)
            
        if cycle is None:
            break
        
        # Verify and remove
        cycle_edges = []
        valid = True
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i+1) % len(cycle)]
            e = (min(u,v), max(u,v))
            if e not in remaining_edges:
                valid = False
                break
            cycle_edges.append(e)
        
        if not valid or len(cycle) < 3:
            # If verification fails, try DFS again or break
            # If greedy returned an invalid cycle, it's a bug. 
            # If DFS returned invalid, it's a bug.
            # Let's just break to be safe, or try to find another cycle?
            # For now, break.
            break
        
        for e in cycle_edges:
            remaining_edges.discard(e)
        
        pieces.append(cycle)
    
    # Remaining edges
    for u, v in remaining_edges:
        pieces.append([u, v])
    
    return pieces
# EVOLVE-BLOCK-END
