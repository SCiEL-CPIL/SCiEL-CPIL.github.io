"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _rnd

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _decompose(edges, deadline, rng):
    if not edges:
        return []
    
    # Build adjacency and degree tracking
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    
    edge_set = set(edges)
    remaining = set(edges)
    pieces = []
    
    # Helper: find a cycle starting from a given edge using DFS
    def find_cycle(start_edge):
        """Find a cycle containing start_edge using randomized DFS."""
        u0, v0 = start_edge
        # Try to build a path from u0 to v0 avoiding start_edge
        # BFS/DFS from u0 to v0 in graph without start_edge
        target = v0
        start = u0
        
        # Randomized DFS to find path from start to target
        visited = set()
        parent = {}
        stack = [start]
        visited.add(start)
        found = False
        
        # Shuffle neighbors for randomness
        while stack:
            node = stack.pop()
            if node == target and node != start:
                # Found path, reconstruct
                path = [node]
                while node != start:
                    node = parent[node]
                    path.append(node)
                path.reverse()
                if len(path) >= 3:
                    return path
                break
            neighbors = list(adj.get(node, []))
            # Filter out edges not in remaining
            valid_neighbors = [nb for nb in neighbors if (min(node,nb), max(node,nb)) in remaining and nb != start or (min(node,nb), max(node,nb)) != start_edge]
            # Actually, we need to avoid using start_edge in the path
            valid_neighbors = [nb for nb in neighbors if (min(node,nb), max(node,nb)) in remaining and (min(node,nb), max(node,nb)) != start_edge]
            rng.shuffle(valid_neighbors)
            for nb in valid_neighbors:
                if nb not in visited:
                    visited.add(nb)
                    parent[nb] = node
                    stack.append(nb)
        
        return None
    
    # Greedy: repeatedly find cycles, prefer edges in high-degree areas
    # Sort edges by some heuristic to find longer cycles first
    # We'll iterate multiple passes
    
    # Strategy: keep trying to find cycles until no more can be found
    # For each remaining edge, try to find a cycle containing it
    # Prioritize edges where both endpoints have high remaining degree
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Score each remaining edge: prefer edges where both endpoints have high degree
        # This helps find longer cycles in dense regions
        def edge_score(e):
            u, v = e
            du = len(adj.get(u, set()))
            dv = len(adj.get(v, set()))
            return du + dv
        
        # Get edges sorted by score (descending) with some randomness
        edge_list = list(remaining)
        # Shuffle then sort by score to break ties randomly
        rng.shuffle(edge_list)
        edge_list.sort(key=edge_score, reverse=True)
        
        cycle_found = False
        for edge in edge_list:
            if time.time() > deadline:
                break
            if edge not in remaining:
                continue
            # Try to find a cycle containing this edge
            cycle = find_cycle(edge)
            if cycle and len(cycle) >= 3:
                # Verify all edges in cycle are in remaining
                valid = True
                for i in range(len(cycle)):
                    a = cycle[i]
                    b = cycle[(i+1) % len(cycle)]
                    e = (min(a,b), max(a,b))
                    if e not in remaining:
                        valid = False
                        break
                if valid:
                    # Remove all edges in cycle
                    for i in range(len(cycle)):
                        a = cycle[i]
                        b = cycle[(i+1) % len(cycle)]
                        e = (min(a,b), max(a,b))
                        remaining.discard(e)
                        # Update adjacency
                        adj[a].discard(b)
                        adj[b].discard(a)
                    pieces.append(cycle)
                    cycle_found = True
                    break
        
        if not cycle_found:
            # No more cycles can be found; take the best remaining edge
            # Just take all remaining edges as single-edge pieces
            # But try one more time with a different strategy
            # For now, just add remaining edges
            for e in sorted(remaining):
                pieces.append([e[0], e[1]])
            remaining.clear()
            break
    
    return pieces

def construct():
    deadline = time.time() + 120  # Safety margin
    rng = _rnd.Random(42)
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        edges = _instance(*spec)
        pieces = _decompose(edges, deadline, rng)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
