"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from collections import defaultdict, deque
import random

def construct():
    # Every edge as its own path; the instance generator is copied from the verifier.
    from random import Random
    
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def path_cover(edges, n):
        """
        Find edge-disjoint paths covering all edges, minimizing the number of paths.
        Uses a greedy approach: repeatedly find longest paths.
        """
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        # Track available edges (as frozenset or tuple)
        available_edges = set()
        for u, v in edges:
            available_edges.add(tuple(sorted((u, v))))
        
        paths = []
        
        while available_edges:
            # Find a longest path using DFS/BFS heuristic
            # Start from a random available edge
            start_edge = next(iter(available_edges))
            start_u, start_v = start_edge
            
            # Try both directions
            best_path = None
            for start, end in [(start_u, start_v), (start_v, start_u)]:
                # Greedy longest path from start
                path = [start]
                visited_nodes = {start}
                current = start
                used_edges_in_path = set()
                
                # We need to track which edges we've used in this path
                # and ensure we only use available edges
                while True:
                    next_node = None
                    next_edge = None
                    for neighbor in adj[current]:
                        edge_key = tuple(sorted((current, neighbor)))
                        if edge_key in available_edges and edge_key not in used_edges_in_path:
                            # Prefer nodes with fewer remaining options (Warnsdorff-like)
                            remaining_options = sum(
                                1 for nb in adj[neighbor]
                                if tuple(sorted((neighbor, nb))) in available_edges
                                and tuple(sorted((neighbor, nb))) not in used_edges_in_path
                            )
                            if next_node is None or remaining_options < best_remaining:
                                next_node = neighbor
                                next_edge = edge_key
                                best_remaining = remaining_options
                            elif remaining_options == best_remaining:
                                # Tie-break: prefer unvisited nodes
                                if neighbor not in visited_nodes and next_node in visited_nodes:
                                    next_node = neighbor
                                    next_edge = edge_key
                                    best_remaining = remaining_options
                    
                    if next_node is None:
                        break
                    
                    path.append(next_node)
                    visited_nodes.add(next_node)
                    used_edges_in_path.add(next_edge)
                    current = next_node
                
                if best_path is None or len(path) > len(best_path):
                    best_path = path
            
            if best_path is None or len(best_path) < 2:
                # Fallback: just take the start edge
                best_path = [start_u, start_v]
            
            # Remove edges used in this path
            for i in range(len(best_path) - 1):
                u, v = best_path[i], best_path[i + 1]
                edge_key = tuple(sorted((u, v)))
                if edge_key in available_edges:
                    available_edges.remove(edge_key)
            
            paths.append(best_path)
        
        return paths
    
    results = []
    for n, p, seed in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(n, p, seed)
        paths = path_cover(edges, n)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
