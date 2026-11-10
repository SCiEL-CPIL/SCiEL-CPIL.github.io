"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def find_cycle(adj, remaining_edges):
        """
        Find a cycle in the graph using DFS.
        adj: dict of vertex -> set of neighbors
        remaining_edges: set of edges (frozenset or tuple) that are still available
        Returns a list of vertices forming the cycle, or None.
        """
        if not remaining_edges:
            return None
        
        # Build adjacency from remaining edges
        adj_remaining = defaultdict(set)
        for u, v in remaining_edges:
            adj_remaining[u].add(v)
            adj_remaining[v].add(u)
        
        visited = {}
        parent = {}
        
        for start in sorted(adj_remaining.keys()):
            if start in visited:
                continue
            # BFS/DFS to find a cycle
            stack = [start]
            visited[start] = True
            parent[start] = None
            found_cycle = None
            
            while stack:
                node = stack[-1]
                if node not in adj_remaining or not adj_remaining[node]:
                    stack.pop()
                    continue
                
                for neighbor in sorted(adj_remaining[node]):
                    if neighbor == parent.get(node, -1):
                        continue
                    if neighbor not in visited:
                        visited[neighbor] = True
                        parent[neighbor] = node
                        stack.append(neighbor)
                        break
                    elif neighbor != parent.get(node, -1):
                        # Found a back edge, extract cycle
                        cycle = [node]
                        curr = node
                        while curr != neighbor:
                            curr = parent[curr]
                            cycle.append(curr)
                        cycle.reverse()
                        found_cycle = cycle
                        break
                if found_cycle:
                    break
                if not stack:
                    break
            
            if found_cycle:
                return found_cycle
        
        return None

    def decompose_graph(edges):
        """
        Decompose the edge set into cycles and single edges, minimizing piece count.
        Strategy: repeatedly find cycles and remove their edges, then leftover edges are single pieces.
        """
        if not edges:
            return []
        
        remaining = set(frozenset(e) for e in edges)
        pieces = []
        
        while remaining:
            # Build adjacency
            adj = defaultdict(set)
            for e in remaining:
                u, v = tuple(e)
                adj[u].add(v)
                adj[v].add(u)
            
            cycle = find_cycle(adj, remaining)
            if cycle and len(cycle) >= 3:
                # Verify all edges in cycle exist in remaining
                valid = True
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    edge = frozenset((u, v))
                    if edge not in remaining:
                        valid = False
                        break
                if valid:
                    pieces.append(cycle)
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        remaining.discard(frozenset((u, v)))
                else:
                    # Fallback: take a single edge
                    e = next(iter(remaining))
                    pieces.append([e[0], e[1]])
                    remaining.discard(e)
            else:
                # No cycle found, take a single edge
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
            
            if time.time() > deadline:
                break
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
