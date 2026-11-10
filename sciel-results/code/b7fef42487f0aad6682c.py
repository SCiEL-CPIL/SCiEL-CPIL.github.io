"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        # Build adjacency
        adj = defaultdict(set)
        edge_set = set()
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
            edge_set.add((min(u, v), max(u, v)))
        
        n = max(max(u, v) for u, v in edges) + 1 if edges else 0
        all_edges = list(edge_set)
        
        if not all_edges:
            return []
        
        # Track which edges are still available
        remaining = set(all_edges)
        pieces = []
        
        # Strategy: repeatedly find long cycles in the remaining graph
        # Use a combination of heuristics:
        # 1. Find cycles via DFS with backtracking (limited)
        # 2. Greedy: find cycles by walking from high-degree vertices
        
        def find_cycle_in_remaining(start=None, max_depth=50):
            """Find a cycle in the remaining edges using DFS."""
            if not remaining:
                return None
            
            # Try to find a cycle starting from each vertex
            candidates = list(remaining)
            random.shuffle(candidates)
            
            # Try multiple starting points
            starts = []
            for u, v in candidates:
                starts.append(u)
                starts.append(v)
            random.shuffle(starts)
            
            for s in starts[:30]:  # limit attempts
                cycle = _dfs_find_cycle(s, remaining, adj, max_depth)
                if cycle is not None:
                    return cycle
            return None
        
        def _dfs_find_cycle(start, rem_edges, adj_map, max_depth):
            """DFS to find a cycle back to start."""
            # We need a path from some neighbor of start back to start
            # such that all edges in the path are in rem_edges and distinct
            
            # Get neighbors of start that have an edge to start in rem_edges
            s_neighbors = [v for v in adj_map[start] if (min(start, v), max(start, v)) in rem_edges]
            if not s_neighbors:
                return None
            
            random.shuffle(s_neighbors)
            
            for neighbor in s_neighbors:
                # Try to find a path from neighbor back to start using remaining edges
                # (excluding the edge start-neighbor for now, we'll add it at the end)
                path = _find_path(neighbor, start, rem_edges, adj_map, 
                                  {start: None}, max_depth, 
                                  exclude_edge=(min(start, neighbor), max(start, neighbor)))
                if path is not None:
                    # path is [neighbor, ..., start]
                    cycle = [start] + path[:-1]  # [start, neighbor, ..., last_before_start]
                    if len(cycle) >= 3:
                        return cycle
            return None
        
        def _find_path(current, target, rem_edges, adj_map, parent_map, max_depth, exclude_edge):
            """Find a path from current to target using BFS/DFS with edge tracking."""
            if current == target:
                # Reconstruct path
                path = [current]
                while parent_map[current] is not None:
                    current = parent_map[current]
                    path.append(current)
                path.reverse()
                return path
            
            if max_depth <= 0:
                return None
            
            neighbors = list(adj_map[current])
            random.shuffle(neighbors)
            
            for nb in neighbors:
                e = (min(current, nb), max(current, nb))
                if e not in rem_edges:
                    continue
                if e == exclude_edge:
                    continue
                if nb in parent_map and parent_map[nb] is not None:
                    continue  # already visited
                
                new_parent = dict(parent_map)
                new_parent[nb] = current
                
                result = _find_path(nb, target, rem_edges, adj_map, new_parent, 
                                     max_depth - 1, exclude_edge)
                if result is not None:
                    return result
            
            return None
        
        # Greedy cycle extraction with multiple strategies
        best_pieces = None
        best_count = float('inf')
        
        # Try multiple random orderings
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            remaining_copy = set(all_edges)
            pieces_copy = []
            
            # Repeatedly extract cycles
            while remaining_copy:
                if time.time() > deadline:
                    break
                
                # Find a cycle
                cycle = None
                
                # Strategy 1: Try to find a cycle using a greedy walk
                cycle = _greedy_cycle(remaining_copy, adj)
                
                # Strategy 2: If that fails, try DFS-based
                if cycle is None:
                    cycle = find_cycle_in_remaining(max_depth=40)
                
                if cycle is None:
                    # No more cycles; add all remaining edges as single pieces
                    for e in remaining_copy:
                        pieces_copy.append([e[0], e[1]])
                    break
                else:
                    # Remove cycle edges from remaining
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        e = (min(u, v), max(u, v))
                        if e in remaining_copy:
                            remaining_copy.remove(e)
                    pieces_copy.append(cycle)
            
            count = len(pieces_copy)
            if count < best_count:
                best_count = count
                best_pieces = pieces_copy
            
            if time.time() > deadline:
                break
        
        if best_pieces is None:
            # Fallback: all edges as single pieces
            best_pieces = [[e[0], e[1]] for e in all_edges]
        
        return best_pieces
    
    def _greedy_cycle(remaining_edges, adj):
        """Greedy: walk to find a cycle."""
        if not remaining_edges:
            return None
        
        # Start from a random edge
        edges_list = list(remaining_edges)
        random.shuffle(edges_list)
        
        for start_edge in edges_list[:20]:
            u, v = start_edge
            # Try to build a cycle starting with u->v
            path = [u, v]
            visited_edges = {(min(u, v), max(u, v))}
            current = v
            
            while current != u:
                # Find neighbors of current that have an edge in remaining
                neighbors = []
                for nb in adj[current]:
                    e = (min(current, nb), max(current, nb))
                    if e in remaining_edges and e not in visited_edges:
                        # Don't go back to previous node unless it's u and we can close
                        if len(path) >= 3 and nb == path[-2]:
                            continue
                        neighbors.append(nb)
                
                if not neighbors:
                    break
                
                # Prefer neighbors that are u (to close cycle) if possible
                if u in neighbors and len(path) >= 3:
                    current = u
                    break
                
                # Otherwise pick a random neighbor (or one that leads toward u)
                # Prefer unvisited nodes
                unvisited = [nb for nb in neighbors if nb not in path]
                if unvisited:
                    random.shuffle(unvisited)
                    current = unvisited[0]
                else:
                    random.shuffle(neighbors)
                    current = neighbors[0]
                
                path.append(current)
                prev = path[-2]
                visited_edges.add((min(prev, current), max(prev, current)))
                
                if len(path) > 30:  # limit cycle length for speed
                    break
            
            if current == u and len(path) >= 3:
                # Verify all edges are in remaining
                valid = True
                for i in range(len(path)):
                    a = path[i]
                    b = path[(i + 1) % len(path)]
                    e = (min(a, b), max(a, b))
                    if e not in remaining_edges:
                        valid = False
                        break
                if valid:
                    return path
        
        return None
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Not enough time; return trivial solution for remaining
            edges = instance(*spec)
            results.append([[e[0], e[1]] for e in edges])
            continue
        
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
