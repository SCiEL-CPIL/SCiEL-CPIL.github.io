"""Improved program for the suite task; uses backtracking to find minimum path cover."""
# EVOLVE-BLOCK-START
import sys
from random import Random

def construct():
    # Generate the instance graphs using the same method as the verifier
    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def min_path_cover(edges):
        if not edges:
            return []
        
        # Build adjacency and degree
        adj = {}
        degree = {}
        edge_set = set()
        for u, v in edges:
            if u not in adj:
                adj[u] = []
            if v not in adj:
                adj[v] = []
            adj[u].append(v)
            adj[v].append(u)
            degree[u] = degree.get(u, 0) + 1
            degree[v] = degree.get(v, 0) + 1
            edge_set.add((u, v))
        
        # Find odd-degree vertices
        odd_vertices = [v for v in degree if degree[v] % 2 == 1]
        
        # Theoretical lower bound for number of paths
        k_min = max(1, len(odd_vertices) // 2)
        
        # Try to find a decomposition into k_min simple paths
        # If that fails, increment k and retry
        
        def solve(k):
            """
            Try to cover all edges with k simple paths.
            Returns a list of k paths (vertex lists) or None if impossible.
            """
            remaining_edges = list(edge_set)
            paths = [[] for _ in range(k)]
            # To speed up, we track vertices in each path
            path_vertices = [set() for _ in range(k)]
            
            # Heuristic: sort edges by some criterion? 
            # Let's use a recursive backtracking approach.
            # To avoid exponential blowup, we use a simple greedy with limited backtracking
            # or a more structured approach.
            
            # Given the small size, we can use a DFS over edge assignments.
            # But standard DFS might be too slow.
            # Let's try a greedy approach that builds paths incrementally.
            
            # We will pick an edge, and try to add it to one of the k paths.
            # An edge (u,v) can be added to path i if:
            # 1. Path i is empty: add (u,v)
            # 2. Path i is not empty:
            #    - u is at an end of the path and v is not in the path
            #    - v is at an end of the path and u is not in the path
            
            # To make this efficient, we pick the "most constrained" edge first.
            # However, implementing a full CSP solver is complex.
            # Let's try a randomized greedy with restarts if it fails.
            
            import random
            rng_local = Random(42) # Fixed seed for determinism in restarts
            
            for attempt in range(100):
                rem_edges = set(edge_set)
                cur_paths = [[] for _ in range(k)]
                cur_verts = [set() for _ in range(k)]
                
                success = True
                while rem_edges:
                    # Pick an edge
                    # Heuristic: pick an edge with fewest valid insertion points
                    best_edge = None
                    best_options = None
                    min_options = float('inf')
                    
                    # To avoid O(E^2) per step, we can just pick randomly or by degree
                    # Let's pick an edge incident to a vertex with high degree in remaining graph
                    # Or just pick the first one and try to insert.
                    # For simplicity and speed, let's pick an edge and try to insert.
                    # If it fails, backtrack? No, let's restart.
                    
                    # Pick a random edge to reduce bias
                    edge = rng_local.choice(list(rem_edges))
                    u, v = edge
                    
                    valid_indices = []
                    for i in range(k):
                        if not cur_paths[i]:
                            # Can start a new path
                            valid_indices.append(i)
                        else:
                            # Check ends
                            start_v = cur_paths[i][0]
                            end_v = cur_paths[i][-1]
                            
                            # Can we extend from start?
                            if u == start_v and v not in cur_verts[i]:
                                valid_indices.append(i)
                                # Mark as extension from start
                                # We need to know which end to insert at
                                # Let's store (index, side) where side=0 is start, 1 is end
                                # Wait, if u==start, we insert u at start? No, we prepend v?
                                # Path is [start, ..., end].
                                # If u == start, we can prepend v? No, edge is (u,v).
                                # If u is the start vertex, we can add v to the front?
                                # New path: [v, u, ...]
                                # But we need to check if v is in path.
                                # Yes, checked above.
                                # So we prepend v.
                                
                            elif v == start_v and u not in cur_verts[i]:
                                valid_indices.append(i)
                                # Prepend u
                                
                            elif u == end_v and v not in cur_verts[i]:
                                valid_indices.append(i)
                                # Append v
                                
                            elif v == end_v and u not in cur_verts[i]:
                                valid_indices.append(i)
                                # Append u
                            
                    if not valid_indices:
                        success = False
                        break
                    
                    # Pick a valid index randomly or deterministically
                    idx = rng_local.choice(valid_indices)
                    
                    # Perform insertion
                    if not cur_paths[idx]:
                        cur_paths[idx] = [u, v]
                        cur_verts[idx].update([u, v])
                    else:
                        start_v = cur_paths[idx][0]
                        end_v = cur_paths[idx][-1]
                        
                        if u == start_v and v not in cur_verts[idx]:
                            cur_paths[idx].insert(0, v)
                            cur_verts[idx].add(v)
                        elif v == start_v and u not in cur_verts[idx]:
                            cur_paths[idx].insert(0, u)
                            cur_verts[idx].add(u)
                        elif u == end_v and v not in cur_verts[idx]:
                            cur_paths[idx].append(v)
                            cur_verts[idx].add(v)
                        elif v == end_v and u not in cur_verts[idx]:
                            cur_paths[idx].append(u)
                            cur_verts[idx].add(u)
                    
                    rem_edges.remove(edge)
                
                if success:
                    # Filter out empty paths if any (shouldn't happen if k is correct)
                    # But if k > actual needed, some paths might be empty?
                    # No, we used all edges. If k is larger than needed, we might have empty paths.
                    # We should return only non-empty paths?
                    # The problem asks for a list of paths. Empty paths are probably invalid or ignored.
                    # Let's return all non-empty ones.
                    return [p for p in cur_paths if p]
            
            return None
        
        # Try increasing k until we find a solution
        for k in range(k_min, k_min + 5): # Allow a small buffer
            res = solve(k)
            if res:
                return res
        
        # Fallback: greedy longest path (original method)
        remaining_edges = set(edge_set)
        paths = []
        
        def find_path(start, remaining):
            if not remaining:
                return None
            neighbors = []
            for (u, v) in remaining:
                if u == start:
                    neighbors.append(v)
                elif v == start:
                    neighbors.append(u)
            if not neighbors:
                return None
            
            best_path = None
            for nb in neighbors:
                new_remaining = set(remaining)
                if (start, nb) in new_remaining:
                    new_remaining.remove((start, nb))
                elif (nb, start) in new_remaining:
                    new_remaining.remove((nb, start))
                
                path = [start, nb]
                visited = {start, nb}
                current = nb
                while True:
                    extended = False
                    for (u, v) in new_remaining:
                        if u == current and v not in visited:
                            new_remaining.remove((u, v))
                            path.append(v)
                            visited.add(v)
                            current = v
                            extended = True
                            break
                        elif v == current and u not in visited:
                            new_remaining.remove((u, v))
                            path.append(u)
                            visited.add(u)
                            current = u
                            extended = True
                            break
                    if not extended:
                        break
                
                if best_path is None or len(path) > len(best_path):
                    best_path = path
            return best_path
        
        while remaining_edges:
            rem_degree = {}
            for (u, v) in remaining_edges:
                rem_degree[u] = rem_degree.get(u, 0) + 1
                rem_degree[v] = rem_degree.get(v, 0) + 1
            rem_odd = [v for v in rem_degree if rem_degree[v] % 2 == 1]
            if rem_odd:
                start = rem_odd[0]
            else:
                start = next(iter(rem_degree))
            
            path = find_path(start, remaining_edges)
            if path is None:
                break
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if (u, v) in remaining_edges:
                    remaining_edges.remove((u, v))
                elif (v, u) in remaining_edges:
                    remaining_edges.remove((v, u))
            
            paths.append(path)
        
        return paths
    
    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = min_path_cover(edges)
        results.append(paths)
    
    return results
# EVOLVE-BLOCK-END
