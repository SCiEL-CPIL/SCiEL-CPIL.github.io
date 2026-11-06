"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    def instance(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        """Decompose edges into minimum number of cycles and single edges."""
        if not edges:
            return []
        
        # Build adjacency
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        remaining = set(edges)
        pieces = []
        
        # Greedy: repeatedly find longest cycle in remaining graph
        # Use randomized DFS with backtracking to find long cycles
        rng = random.Random(42)
        
        while remaining:
            if time.time() > deadline - 10:
                # Time's up, return what we have
                break
            
            # Find a cycle in the remaining graph
            # Try multiple random starting points and DFS strategies
            best_cycle = None
            best_len = 0
            
            # Get all vertices in remaining graph
            verts_in_rem = set()
            for u, v in remaining:
                verts_in_rem.add(u)
                verts_in_rem.add(v)
            
            # Try random orderings to find longer cycles
            for attempt in range(min(20, len(verts_in_rem))):
                if time.time() > deadline - 5:
                    break
                
                # Random start vertex
                start = rng.choice(list(verts_in_rem))
                
                # DFS to find a cycle back to start
                # Use iterative DFS with path tracking
                stack = [(start, [start], set([start]))]
                found = False
                
                while stack and not found:
                    if time.time() > deadline - 2:
                        break
                    node, path, visited = stack.pop()
                    
                    # Try neighbors in random order
                    neighbors = list(adj.get(node, []))
                    # Only consider neighbors connected by remaining edges
                    valid_neighbors = []
                    for nb in neighbors:
                        edge = tuple(sorted((node, nb)))
                        if edge in remaining:
                            valid_neighbors.append(nb)
                    
                    rng.shuffle(valid_neighbors)
                    
                    for nb in reversed(valid_neighbors):  # reversed because we pop
                        if nb == start and len(path) >= 3:
                            # Found a cycle!
                            cycle = path[:]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle
                            found = True
                            break
                        elif nb not in visited:
                            new_visited = visited | {nb}
                            new_path = path + [nb]
                            stack.append((nb, new_path, new_visited))
                
                if found:
                    break
            
            if best_cycle:
                # Remove edges in the cycle from remaining
                for i in range(len(best_cycle)):
                    u = best_cycle[i]
                    v = best_cycle[(i + 1) % len(best_cycle)]
                    edge = tuple(sorted((u, v)))
                    if edge in remaining:
                        remaining.discard(edge)
                pieces.append(best_cycle[:])
            else:
                # No cycle found; take one edge as a single-edge piece
                # Pick an edge, prefer ones with higher degree endpoints
                edge = None
                max_deg = -1
                for u, v in remaining:
                    deg = len(adj.get(u, [])) + len(adj.get(v, []))
                    if deg > max_deg:
                        max_deg = deg
                        edge = (u, v)
                if edge:
                    remaining.discard(edge)
                    pieces.append(list(edge))
                else:
                    break
        
        # Add any remaining edges as single-edge pieces
        for edge in remaining:
            if time.time() > deadline - 1:
                break
            pieces.append(list(edge))
        
        return pieces

    results = []
    for spec in ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13)):
        if time.time() > deadline - 5:
            # Not enough time, return all edges as pieces
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
