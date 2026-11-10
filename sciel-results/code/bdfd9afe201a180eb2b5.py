"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    """
    Decompose each instance graph into edge-disjoint cycles and single edges,
    minimizing the total number of pieces (cycles + single edges).
    
    Strategy:
    1. Generate the instance graphs.
    2. For each graph, use a heuristic to find many edge-disjoint cycles.
       - Repeatedly find a long cycle in the remaining graph using a greedy/DFS approach.
       - Remove the cycle's edges from the graph.
       - Repeat until no more cycles can be found (or time runs out).
    3. The remaining edges become single-edge pieces.
    4. Return the list of pieces for each graph.
    """
    start_time = time.time()
    deadline = 130.0  # 130s to be safe within 120s limit (actually 150s wall-clock, so 130 is safe)
    
    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)
    
    def find_cycle(adj, n):
        """
        Find a simple cycle in the graph represented by adj (adjacency set).
        Returns a list of vertices forming the cycle, or None if no cycle exists.
        Uses a DFS-based approach to find a long cycle.
        """
        # Try to find a long cycle by starting from different vertices
        best_cycle = None
        for start in range(n):
            if not adj[start]:
                continue
            # DFS to find a cycle back to start
            visited = {start}
            path = [start]
            stack = [(start, iter(sorted(adj[start])))]
            found = False
            while stack:
                node, neighbors = stack[-1]
                try:
                    next_node = next(neighbors)
                    if next_node == start and len(path) >= 3:
                        # Found a cycle
                        cycle = path[:]
                        if best_cycle is None or len(cycle) > len(best_cycle):
                            best_cycle = cycle
                        found = True
                        break
                    elif next_node not in visited:
                        visited.add(next_node)
                        path.append(next_node)
                        stack.append((next_node, iter(sorted(adj[next_node]))))
                    # else: skip visited node
                except StopIteration:
                    stack.pop()
                    if len(path) > 0:
                        path.pop()
                        visited.discard(node)
            if found:
                break  # We found a cycle, but let's try to get a longer one? 
                # Actually, for efficiency, let's just return the first decent cycle.
                # Or we can try a few starts.
        
        # If the above didn't find a good one, try a simpler approach:
        # Find any cycle using a union-find or just DFS for any cycle.
        if best_cycle is None:
            # Use a simpler cycle finding: for each edge (u,v), check if u and v are in same component of G - (u,v)
            # This is O(m * (n+m)) which is fine for n=30.
            # But let's try a greedy DFS for any cycle first.
            for u in range(n):
                if not adj[u]:
                    continue
                visited = {u}
                parent = {u: -1}
                stack = [u]
                found = False
                while stack and not found:
                    node = stack.pop()
                    for v in adj[node]:
                        if v == u and len(stack) + 1 >= 3:
                            # Reconstruct path
                            cycle = [node]
                            cur = node
                            while cur != u:
                                cur = parent[cur]
                                if cur == -1:
                                    break
                                cycle.append(cur)
                            cycle.append(u)
                            if best_cycle is None or len(cycle) > len(best_cycle):
                                best_cycle = cycle
                            found = True
                            break
                        elif v not in visited:
                            visited.add(v)
                            parent[v] = node
                            stack.append(v)
                if found:
                    break
        
        return best_cycle

    def decompose(edges, n):
        """
        Decompose the graph into cycles and single edges.
        Returns a list of pieces.
        """
        # Build adjacency structure
        adj = defaultdict(set)
        remaining_edges = set()
        for e in edges:
            u, v = e
            adj[u].add(v)
            adj[v].add(u)
            remaining_edges.add(tuple(sorted(e)))
        
        pieces = []
        
        # Repeatedly find and remove cycles
        while True:
            if time.time() > deadline - 5:
                break
            
            # Find a cycle
            cycle = find_cycle(adj, n)
            if cycle is None:
                break
            
            # Remove the cycle's edges
            for i in range(len(cycle) - 1):
                u, v = cycle[i], cycle[i + 1]
                edge = tuple(sorted((u, v)))
                if edge in remaining_edges:
                    remaining_edges.remove(edge)
                    adj[u].remove(v)
                    adj[v].remove(u)
            
            # Add the cycle as a piece
            # The cycle is given as a list of vertices, closing edge implied
            # We need to make sure it's a valid cycle representation
            pieces.append(cycle)
        
        # Remaining edges become single-edge pieces
        for e in remaining_edges:
            pieces.append(list(e))
        
        return pieces

    results = []
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    
    for spec in specs:
        if time.time() > deadline - 5:
            # Not enough time, return trivial solution for remaining
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        
        edges = instance(*spec)
        n = spec[0]
        pieces = decompose(edges, n)
        results.append(pieces)
    
    return results

# EVOLVE-BLOCK-END
