"""Decompose random graphs into cycles and edges to minimize piece count."""
import time
import random
from random import Random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_random_path(start, end, edges, rng):
        """Find a random simple path from start to end in the graph defined by edges."""
        if start == end:
            return [start]
        
        # Build adjacency for current edges
        adj = {}
        for u, v in edges:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
            
        if start not in adj or end not in adj:
            return None
            
        # DFS with random neighbor ordering
        visited = {start}
        path = [start]
        
        def dfs(node):
            if node == end:
                return True
            neighbors = list(adj.get(node, []))
            # Filter visited
            neighbors = [n for n in neighbors if n not in visited]
            rng.shuffle(neighbors)
            for n in neighbors:
                visited.add(n)
                path.append(n)
                if dfs(n):
                    return True
                path.pop()
                visited.discard(n)
            return False
            
        if dfs(start):
            return path[:]
        return None

    def decompose(edges):
        """Decompose a set of edges into cycles and single edges, minimizing piece count."""
        if not edges:
            return []

        best_pieces = None
        # Allocate time per graph. Total 140s / 3 graphs ~ 46s each.
        # We will use a loop until time runs out, keeping the best result.
        graph_deadline = time.time() + 45
        
        while time.time() < graph_deadline:
            current_edges = set(edges)
            pieces = []
            rng = Random() # Use system random for restarts
            
            while current_edges:
                # Pick a random edge to try to extend into a cycle
                edge = next(iter(current_edges))
                u, v = edge
                current_edges.discard(edge)
                
                # Try to find a long path from u to v in current_edges
                # We use a randomized DFS to find a simple path.
                # We try a few times to find a longer path.
                best_path = None
                # Limit attempts to keep it fast, but enough to find long paths
                for _ in range(20):
                    path = find_random_path(u, v, current_edges, rng)
                    if path:
                        if len(path) >= 3: # Must be at least 3 vertices for a valid cycle
                            if best_path is None or len(path) > len(best_path):
                                best_path = path
                            # We could break early if we find a very long path
                            # But let's just take the best of the 20
                
                if best_path:
                    # Path is [u, ..., v]
                    # Cycle is defined by these vertices
                    # Remove path edges from current_edges
                    for i in range(len(best_path) - 1):
                        e1, e2 = best_path[i], best_path[i+1]
                        e = (min(e1, e2), max(e1, e2))
                        current_edges.discard(e)
                    
                    pieces.append(best_path)
                else:
                    # No valid cycle path found, use single edge
                    pieces.append([u, v])
            
            if best_pieces is None or len(pieces) < len(best_pieces):
                best_pieces = pieces
                # If we found a perfect decomposition (all cycles?), we could break
                # But single edges are allowed.
        
        if best_pieces is None:
            # Fallback: trivial decomposition
            return [list(e) for e in edges]
        
        return best_pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time; use trivial decomposition
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        result = decompose(edges)
        results.append(result)

    return results
