import time
from random import Random
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    DEADLINE = 100.0  # Safety margin

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    def decompose_graph(edges):
        if not edges:
            return []
        
        # Build adjacency list
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Track used edges
        used_edges = set()
        paths = []
        
        # Helper to check if graph is empty
        def has_edges():
            return len(used_edges) < len(edges)
        
        # Helper to get remaining edges
        def get_remaining_edges():
            return [e for e in edges if e not in used_edges]
            
        # Helper to compute degrees in remaining graph
        def get_degrees():
            deg = defaultdict(int)
            for u, v in edges:
                if (u, v) not in used_edges:
                    deg[u] += 1
                    deg[v] += 1
            return deg
            
        # Beam Search for longest path
        def find_longest_path(start):
            # State: (current_node, path_tuple)
            # We use a list of tuples to represent the beam
            # Score: length of path
            beam = [(start, (start,))]
            best_path = (start,)
            
            while beam:
                next_beam = []
                for curr, path in beam:
                    best_path = path if len(path) > len(best_path) else best_path
                    # Try extending
                    for neighbor in adj[curr]:
                        if neighbor in path:
                            continue
                        # Check if edge (curr, neighbor) is available
                        e = tuple(sorted((curr, neighbor)))
                        if e in used_edges:
                            continue
                        next_beam.append((neighbor, path + (neighbor,)))
                
                if not next_beam:
                    break
                
                # Prune: keep top W paths
                # Sort by length, then maybe by something else?
                # To avoid duplicates in the beam, we might need a set of visited states?
                # But paths are unique by their sequence.
                # However, many paths might end at the same node with same length.
                # Simple pruning: keep unique paths.
                unique_next_beam = set()
                for state in next_beam:
                    unique_next_beam.add(state)
                
                # If too many, sort and truncate
                if len(unique_next_beam) > 50:
                    # Sort by length descending
                    sorted_beam = sorted(unique_next_beam, key=lambda x: len(x[1]), reverse=True)
                    beam = sorted_beam[:50]
                else:
                    beam = list(unique_next_beam)
                    
            return list(best_path)

        while has_edges():
            if time.time() - start_time > DEADLINE:
                break
                
            degs = get_degrees()
            odd_vertices = [v for v, d in degs.items() if d % 2 == 1]
            
            # Choose start vertex
            if odd_vertices:
                # Pick odd vertex with highest degree
                start = max(odd_vertices, key=lambda v: degs[v])
            else:
                # Pick any vertex with degree > 0
                candidates = [v for v, d in degs.items() if d > 0]
                if not candidates:
                    break
                start = candidates[0]
            
            path = find_longest_path(start)
            
            if len(path) < 2:
                # Stuck? Should not happen if has_edges() and connected, but safety
                # Find any edge
                rem_edges = get_remaining_edges()
                if rem_edges:
                    u, v = rem_edges[0]
                    path = [u, v]
                else:
                    break
            
            # Add path to results
            paths.append(path)
            
            # Mark edges as used
            for i in range(len(path) - 1):
                e = tuple(sorted((path[i], path[i+1])))
                used_edges.add(e)
                
        return paths

    results = []
    for spec in ((20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)):
        edges = instance(*spec)
        paths = decompose_graph(edges)
        results.append(paths)
        
        if time.time() - start_time > DEADLINE:
            # Fill remaining with trivial paths if we ran out of time?
            # Better to just return what we have, but we need valid output.
            # The loop above breaks, but we need to ensure we return 3 lists.
            pass
            
    return results
