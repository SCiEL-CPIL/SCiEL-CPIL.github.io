import time
import random
from collections import defaultdict, deque
from random import Random

_DEADLINE = 148.0
_START = time.time()

def _remaining():
    return _DEADLINE - (time.time() - _START)

def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, max_time=10.0):
        if not edges:
            return []
        
        # Precompute adjacency and edge set
        adj = {i: set() for i in range(n)}
        edge_set = set(edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Helper to check if edge exists in a set
        def has_edge(u, v, eset):
            return (u, v) in eset or (v, u) in eset

        def find_longest_path(start, eset, adj_local, visited_set):
            """
            Greedily find a long simple path starting from 'start' using edges in 'eset'.
            Uses a heuristic: prefer neighbors with lower remaining degree (Wielandt's heuristic for Eulerian trails)
            or random if tied.
            """
            path = [start]
            visited = set(visited_set) | {start}
            current = start
            
            while True:
                # Find available neighbors
                available = []
                for nb in adj_local[current]:
                    if nb not in visited and has_edge(current, nb, eset):
                        available.append(nb)
                
                if not available:
                    break
                
                # Heuristic: choose neighbor with minimum degree in the remaining graph
                # This helps create longer trails by avoiding "traps"
                min_deg = float('inf')
                best_nb = None
                for nb in available:
                    deg = 0
                    for x in adj_local[nb]:
                        if x not in visited and has_edge(nb, x, eset):
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_nb = nb
                
                # Randomize slightly to escape local optima
                if random.random() < 0.2 and len(available) > 1:
                    best_nb = random.choice(available)
                
                # Move
                u, v = current, best_nb
                if (u, v) in eset:
                    eset.discard((u, v))
                else:
                    eset.discard((v, u))
                
                path.append(best_nb)
                visited.add(best_nb)
                current = best_nb
                
            return path

        def decompose_with_odd_pairing(edges, n):
            """
            Structural approach:
            1. Identify odd-degree vertices.
            2. Pair them up.
            3. Add virtual edges between pairs.
            4. Find Eulerian circuit in the augmented graph.
            5. Split the circuit at virtual edges to get paths.
            6. If paths are not simple (they might not be, since Eulerian circuits can repeat vertices),
               we need to handle that. However, for small random graphs, the "trails" from Eulerian
               decomposition are often simple or close to it.
               
               Actually, the problem requires SIMPLE paths. An Eulerian circuit is a trail, not a path.
               So this direct approach fails if the Eulerian circuit repeats vertices.
               
               Alternative Structural Approach:
               Use the property that min path cover size is related to odd vertices.
               We can try to construct paths that connect pairs of odd vertices directly.
               
               Let's try a different structural strategy:
               1. Compute odd vertices.
               2. If no odd vertices, we can try to find a Hamiltonian path (unlikely) or just use greedy.
               3. If odd vertices exist, pair them.
               4. For each pair, try to find a simple path between them using remaining edges.
               5. This is hard because paths must be edge-disjoint.
               
               Let's stick to a robust Greedy + Local Search hybrid, but with a better initialization
               inspired by the "longest path" idea from inspiration 1, and the "random restart" from current.
               
               New Strategy:
               1. Greedy construction: Start from an odd vertex (if any) or random vertex.
                  Extend greedily using "minimum degree" heuristic.
               2. When stuck, start a new path.
               3. Repeat with different random seeds/orderings.
               4. Keep the best solution.
            """
            best_paths = None
            best_count = float('inf')
            
            start_time = time.time()
            
            # Multiple restarts with different randomization
            while time.time() - start_time < max_time and _remaining() > 2.0:
                remaining = set(edge_set)
                paths = []
                
                # Determine starting vertices preference
                # Calculate degrees in the full graph
                deg = {i: 0 for i in range(n)}
                for u, v in edges:
                    deg[u] += 1
                    deg[v] += 1
                odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
                
                # We will pick start vertices randomly, but bias towards odd vertices initially
                # to help satisfy the odd-degree endpoints requirement of path covers.
                
                while remaining:
                    # Pick a start vertex
                    # Heuristic: Pick a vertex with high degree in remaining graph?
                    # Or pick an odd vertex?
                    # Let's pick a random edge to start, as in the current program,
                    # but we can try to start from specific vertices.
                    
                    # To improve on current, let's try to start from an odd-degree vertex in the *remaining* graph
                    rem_deg = {i: 0 for i in range(n)}
                    for u, v in remaining:
                        rem_deg[u] += 1
                        rem_deg[v] += 1
                    rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]
                    
                    if rem_odd and random.random() < 0.8:
                        start_v = random.choice(rem_odd)
                    else:
                        # Pick random vertex with remaining edges
                        start_v = random.choice([v for v in rem_deg if rem_deg[v] > 0])
                    
                    # Find a long path starting from start_v
                    path = find_longest_path(start_v, remaining, adj, set())
                    
                    if not path or len(path) < 2:
                        # Fallback: just take one edge
                        # Find any edge incident to start_v
                        found_edge = False
                        for u, v in remaining:
                            if u == start_v:
                                path = [start_v, v]
                                remaining.discard((u, v))
                                found_edge = True
                                break
                            elif v == start_v:
                                path = [start_v, u]
                                remaining.discard((u, v))
                                found_edge = True
                                break
                        if not found_edge:
                            # Should not happen if remaining is not empty and start_v has degree > 0
                            break
                    else:
                        # Remove edges from path
                        for i in range(len(path) - 1):
                            u, v = path[i], path[i+1]
                            if (u, v) in remaining:
                                remaining.discard((u, v))
                            else:
                                remaining.discard((v, u))
                    
                    paths.append(path)
                
                if len(paths) < best_count:
                    best_count = len(paths)
                    best_paths = [p[:] for p in paths]
                    
            if best_paths is None:
                return [[u, v] for u, v in edges]
                
            return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    # Allocate time budget
    # 140s total. 3 graphs. ~45s each?
    # But we have _remaining() guard.
    # Let's give each graph a chunk of time, but check global deadline.
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        # Give more time to later graphs if we are fast?
        # For now, fixed 10s per graph is safe, but we can use up to remaining time.
        # Let's use 10s per graph as a baseline, capped by global deadline.
        max_t = 12.0
        if _remaining() < max_t:
            max_t = max(1.0, _remaining() - 1.0) # Leave 1s margin
            
        paths = decompose_with_odd_pairing(edges, n, max_time=max_t)
        results.append(paths)
        
    return results

# EVOLVE-BLOCK-START
# The construct function is defined above.
# This block is kept for compatibility if the harness expects it, 
# but the logic is in the construct function defined above.
# Actually, the prompt says "Keep the function names and signatures of the seed program."
# The seed program had `construct` defined, then redefined in EVOLVE-BLOCK.
# I will define the new logic in the EVOLVE-BLOCK to follow the pattern, 
# or just ensure `construct` is the final definition.
# Since Python uses the last definition, I will put the main logic in the EVOLVE-BLOCK 
# to be safe and clear.

# Re-defining construct in the evolve block to ensure it's the active one.
def construct():
    def instance_graph(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges, n, max_time=10.0):
        if not edges:
            return []
        
        adj = {i: set() for i in range(n)}
        edge_set = set(edges)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        def has_edge(u, v, eset):
            return (u, v) in eset or (v, u) in eset

        def find_longest_path(start, eset, adj_local, visited_set):
            path = [start]
            visited = set(visited_set) | {start}
            current = start
            
            while True:
                available = []
                for nb in adj_local[current]:
                    if nb not in visited and has_edge(current, nb, eset):
                        available.append(nb)
                
                if not available:
                    break
                
                # Heuristic: minimum degree in remaining graph
                min_deg = float('inf')
                best_nb = None
                for nb in available:
                    deg = 0
                    for x in adj_local[nb]:
                        if x not in visited and has_edge(nb, x, eset):
                            deg += 1
                    if deg < min_deg:
                        min_deg = deg
                        best_nb = nb
                
                if random.random() < 0.2 and len(available) > 1:
                    best_nb = random.choice(available)
                
                u, v = current, best_nb
                if (u, v) in eset:
                    eset.discard((u, v))
                else:
                    eset.discard((v, u))
                
                path.append(best_nb)
                visited.add(best_nb)
                current = best_nb
                
            return path

        best_paths = None
        best_count = float('inf')
        start_time = time.time()
        
        while time.time() - start_time < max_time and _remaining() > 2.0:
            remaining = set(edge_set)
            paths = []
            
            deg = {i: 0 for i in range(n)}
            for u, v in edges:
                deg[u] += 1
                deg[v] += 1
            odd_vertices = [v for v in range(n) if deg[v] % 2 == 1]
            
            while remaining:
                rem_deg = {i: 0 for i in range(n)}
                for u, v in remaining:
                    rem_deg[u] += 1
                    rem_deg[v] += 1
                rem_odd = [v for v in rem_deg if rem_deg[v] > 0 and rem_deg[v] % 2 == 1]
                
                if rem_odd and random.random() < 0.8:
                    start_v = random.choice(rem_odd)
                else:
                    start_v = random.choice([v for v in rem_deg if rem_deg[v] > 0])
                
                path = find_longest_path(start_v, remaining, adj, set())
                
                if not path or len(path) < 2:
                    found_edge = False
                    for u, v in remaining:
                        if u == start_v:
                            path = [start_v, v]
                            remaining.discard((u, v))
                            found_edge = True
                            break
                        elif v == start_v:
                            path = [start_v, u]
                            remaining.discard((u, v))
                            found_edge = True
                            break
                    if not found_edge:
                        break
                else:
                    for i in range(len(path) - 1):
                        u, v = path[i], path[i+1]
                        if (u, v) in remaining:
                            remaining.discard((u, v))
                        else:
                            remaining.discard((v, u))
                
                paths.append(path)
            
            if len(paths) < best_count:
                best_count = len(paths)
                best_paths = [p[:] for p in paths]
                
        if best_paths is None:
            return [[u, v] for u, v in edges]
            
        return best_paths

    specs = ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3))
    results = []
    
    for i, (n, p, seed) in enumerate(specs):
        edges = instance_graph(n, p, seed)
        max_t = 12.0
        if _remaining() < max_t:
            max_t = max(1.0, _remaining() - 1.0)
            
        paths = decompose_graph(edges, n, max_time=max_t)
        results.append(paths)
        
    return results
# EVOLVE-BLOCK-END
