"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

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
        if not edges:
            return []
        
        remaining = set(edges)
        pieces = []
        rng = random.Random(12345)
        
        # Build dynamic adjacency
        def get_adj(rem):
            adj = defaultdict(set)
            for u, v in rem:
                adj[u].add(v)
                adj[v].add(u)
            return adj
        
        # Find longest cycle using greedy path extension with backtracking
        def find_long_cycle(rem, adj, max_attempts=50):
            if not rem:
                return None
            
            verts = set()
            for u, v in rem:
                verts.add(u)
                verts.add(v)
            verts = list(verts)
            if not verts:
                return None
            
            best_cycle = None
            best_len = 0
            
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                start = rng.choice(verts)
                
                # Greedy path extension with some backtracking
                path = [start]
                visited = {start}
                current = start
                
                # Try to extend path greedily, preferring neighbors with fewer options
                extended = True
                while extended:
                    extended = False
                    neighbors = []
                    for nb in adj.get(current, []):
                        edge = tuple(sorted((current, nb)))
                        if edge in rem:
                            if nb == start and len(path) >= 3:
                                # Can close cycle
                                if len(path) > best_len:
                                    best_len = len(path)
                                    best_cycle = path[:]
                                return best_cycle
                            elif nb not in visited:
                                neighbors.append(nb)
                    
                    if not neighbors:
                        break
                    
                    # Prefer neighbors with fewer remaining options (dead-end avoidance)
                    def score(nb):
                        cnt = 0
                        for nn in adj.get(nb, []):
                            e = tuple(sorted((nb, nn)))
                            if e in rem and nn != start and nn not in visited:
                                cnt += 1
                        return cnt
                    
                    neighbors.sort(key=score)
                    # Sometimes pick randomly for diversity
                    if rng.random() < 0.3:
                        next_node = rng.choice(neighbors)
                    else:
                        next_node = neighbors[0]
                    
                    path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                    extended = True
                
                # If we couldn't close, try to find a cycle in the path
                # Check if any vertex in path connects back to an earlier vertex
                # This is expensive, so only do it for longer paths
                if len(path) > best_len and len(path) >= 4:
                    # Try to find a cycle by checking if path[0] connects to any later vertex
                    for i in range(2, len(path)):
                        edge = tuple(sorted((path[0], path[i])))
                        if edge in rem:
                            cycle = path[:i+1]
                            if len(cycle) > best_len:
                                best_len = len(cycle)
                                best_cycle = cycle[:]
                            break
            
            return best_cycle
        
        # Find a 2-factor: set of vertex-disjoint cycles covering a subset of vertices
        def find_2factor(rem, adj):
            """Find vertex-disjoint cycles covering as many edges as possible."""
            if not rem:
                return []
            
            cycles = []
            used_vertices = set()
            working_rem = set(rem)
            
            # Repeatedly find cycles in the remaining graph, avoiding used vertices
            while working_rem:
                if time.time() > deadline - 5:
                    break
                
                # Filter to only consider edges not incident to used vertices
                available_edges = set()
                for u, v in working_rem:
                    if u not in used_vertices and v not in used_vertices:
                        available_edges.add((u, v))
                
                if not available_edges:
                    break
                
                # Find a cycle in available_edges
                cycle = find_long_cycle(available_edges, adj, max_attempts=30)
                
                if cycle:
                    cycles.append(cycle)
                    for v in cycle:
                        used_vertices.add(v)
                    # Remove all edges in this cycle from working_rem
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        working_rem.discard(edge)
                else:
                    break
            
            return cycles
        
        # Main decomposition loop
        while remaining:
            if time.time() > deadline - 3:
                break
            
            adj = get_adj(remaining)
            
            # Strategy 1: Try to find a 2-factor (multiple vertex-disjoint cycles)
            cycles = find_2factor(remaining, adj)
            
            if cycles:
                for cycle in cycles:
                    # Remove cycle edges
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        remaining.discard(edge)
                    pieces.append(cycle[:])
            else:
                # Strategy 2: Find a single long cycle
                cycle = find_long_cycle(remaining, adj, max_attempts=100)
                
                if cycle:
                    for i in range(len(cycle)):
                        u = cycle[i]
                        v = cycle[(i + 1) % len(cycle)]
                        edge = tuple(sorted((u, v)))
                        remaining.discard(edge)
                    pieces.append(cycle[:])
                else:
                    # No cycle found; take a single edge
                    # Pick an edge that seems most "isolated"
                    edge = None
                    min_score = float('inf')
                    for u, v in remaining:
                        deg_u = len(adj.get(u, []))
                        deg_v = len(adj.get(v, []))
                        score = deg_u + deg_v
                        if score < min_score:
                            min_score = score
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
        if time.time() > deadline - 10:
            edges = instance(*spec)
            results.append([list(e) for e in edges])
            continue
        edges = instance(*spec)
        pieces = decompose_graph(edges)
        results.append(pieces)
    
    return results
# EVOLVE-BLOCK-END
