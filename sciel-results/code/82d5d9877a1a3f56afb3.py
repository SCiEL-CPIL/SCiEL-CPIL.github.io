"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
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
    
    def find_cycles_decomposition(edges):
        """Decompose edges into cycles and single edges, minimizing piece count."""
        remaining = set(edges)
        pieces = []
        
        # Build adjacency
        adj = {}
        for u, v in edges:
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        
        # Greedily find cycles: look for shortest cycles in remaining edges
        # Strategy: repeatedly find a cycle, remove its edges, repeat
        # Use a heuristic: for each edge, try to extend to form a cycle
        
        while remaining:
            if time.time() > deadline:
                break
            
            # Try to find a cycle using BFS/DFS from each edge
            cycle_found = None
            
            # Sort edges for deterministic behavior
            for start_edge in sorted(remaining):
                u0, v0 = start_edge
                # Try to find a path from v0 back to u0 using remaining edges
                # excluding the start edge itself
                visited = {u0}
                path = [u0, v0]
                
                # DFS to find path from v0 to u0
                stack = [(v0, [u0, v0])]
                found = False
                while stack and not found:
                    node, cur_path = stack.pop()
                    if node == u0 and len(cur_path) >= 3:
                        # Found a cycle
                        cycle_found = cur_path
                        found = True
                        break
                    for nb in sorted(adj.get(node, [])):
                        if (min(node, nb), max(node, nb)) not in remaining:
                            continue
                        if (min(node, nb), max(node, nb)) == (u0, v0):
                            continue  # skip the start edge for now
                        if nb not in visited:
                            visited.add(nb)
                            stack.append((nb, cur_path + [nb]))
                
                if cycle_found:
                    break
            
            if cycle_found and len(cycle_found) >= 3:
                # Remove cycle edges from remaining
                for i in range(len(cycle_found)):
                    a, b = cycle_found[i], cycle_found[(i + 1) % len(cycle_found)]
                    e = (min(a, b), max(a, b))
                    remaining.discard(e)
                pieces.append(cycle_found)
            else:
                # No more cycles; add all remaining as single edges
                for e in sorted(remaining):
                    pieces.append([e[0], e[1]])
                remaining = set()
        
        # If we hit deadline, add remaining as single edges
        if remaining:
            for e in sorted(remaining):
                pieces.append([e[0], e[1]])
        
        return pieces
    
    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline:
            # Fallback: all edges as single pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
        else:
            edges = instance(*spec)
            pieces = find_cycles_decomposition(edges)
            results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
