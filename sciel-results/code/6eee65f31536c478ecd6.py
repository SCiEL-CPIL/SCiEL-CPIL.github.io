import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = P['n']
    all_vertices = list(range(n))
    
    # Build adjacency and edge set
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u,v), max(u,v))
        
    def is_connected(available_edges):
        """Check if the graph formed by available_edges is connected."""
        if not available_edges:
            return n == 1
        adj_temp = {i: set() for i in range(n)}
        for u, v in available_edges:
            adj_temp[u].add(v)
            adj_temp[v].add(u)
        
        visited = set()
        stack = [0]
        visited.add(0)
        while stack:
            node = stack.pop()
            for nb in adj_temp[node]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        return len(visited) == n

    def min_degree(available_edges):
        """Check if all vertices have degree >= 2 in available_edges."""
        degrees = {i: 0 for i in range(n)}
        for u, v in available_edges:
            degrees[u] += 1
            degrees[v] += 1
        for i in range(n):
            if degrees[i] < 2:
                return False
        return True

    def find_hamiltonian_cycle(available_edges, start=0, time_limit=5.0):
        """Find a Hamiltonian cycle using only edges in available_edges.
        Uses backtracking with improved pruning."""
        t0 = time.time()
        path = [start]
        visited = {start}
        
        # Precompute degree of each vertex in available_edges for fast lookup
        deg_in_avail = {v: 0 for v in range(n)}
        for u, w in available_edges:
            deg_in_avail[u] += 1
            deg_in_avail[w] += 1
        
        # Precompute adjacency within available edges
        avail_adj = {v: set() for v in range(n)}
        for u, w in available_edges:
            avail_adj[u].add(w)
            avail_adj[w].add(u)
        
        def backtrack():
            if time.time() - t0 > time_limit:
                return "timeout"
            
            if len(path) == n:
                last = path[-1]
                if edge_key(last, start) in available_edges:
                    return list(path)
                return None
            
            last = path[-1]
            # Get neighbors of last that are in available_edges and not visited
            candidates = []
            for nb in avail_adj[last]:
                if nb not in visited:
                    candidates.append(nb)
            
            if not candidates:
                return None

            # Heuristic: Sort candidates by their remaining degree in available graph
            # (degree minus 1 if they connect to last, to account for edge usage)
            def sort_key(nb):
                # Count available neighbors not yet visited
                remaining_neighbors = 0
                for nb2 in avail_adj[nb]:
                    if nb2 not in visited and nb2 != last:
                        remaining_neighbors += 1
                # Also consider if nb connects back to start (for closing the cycle)
                if nb == start or edge_key(nb, start) in available_edges:
                    remaining_neighbors += 1
                return remaining_neighbors
            
            candidates.sort(key=sort_key)

            for nb in candidates:
                # Quick pruning: check if nb has at least one other available neighbor
                # (besides last) that is not visited, or can connect to start
                nb_has_exit = False
                for nb2 in avail_adj[nb]:
                    if nb2 != last and nb2 not in visited:
                        nb_has_exit = True
                        break
                if not nb_has_exit and len(path) < n - 1:
                    # nb must be able to connect to start if it's the second-to-last
                    if edge_key(nb, start) not in available_edges:
                        continue
                if not nb_has_exit and len(path) == n - 1:
                    if edge_key(nb, start) not in available_edges:
                        continue

                # Check if removing this edge would leave any unvisited vertex with degree < 2
                valid = True
                for v in range(n):
                    if v in visited and v != nb:
                        continue
                    # Count available edges incident to v, excluding (last, nb)
                    deg_v = 0
                    for u2 in avail_adj[v]:
                        ek = edge_key(last, nb)
                        if edge_key(v, u2) != ek:
                            deg_v += 1
                    if v == nb:
                        # nb will be visited, so it needs at least 1 edge to an unvisited node or start
                        if deg_v < 1 and edge_key(nb, start) not in available_edges:
                            valid = False
                            break
                    elif v not in visited:
                        # Unvisited vertices need degree >= 2
                        if deg_v < 2:
                            valid = False
                            break
                
                if valid:
                    path.append(nb)
                    visited.add(nb)
                    result = backtrack()
                    if result is not None and result != "timeout":
                        return result
                    path.pop()
                    visited.remove(nb)
                    
            return None
        
        result = backtrack()
        if result == "timeout":
            return None
        return result

    cycles = []
    available = set(edge_set)
    
    # Try to find up to 3 cycles
    max_cycles = 3
    for _ in range(max_cycles):
        if time.time() > deadline:
            break
        
        found = None
        
        # Try different starting vertices
        starts_to_try = [0] + list(range(1, n))
        for start in starts_to_try:
            if time.time() > deadline:
                break
            remaining_time = deadline - time.time()
            if remaining_time < 0.5:
                break
            time_limit = min(remaining_time * 0.4, 8.0)
            found = find_hamiltonian_cycle(available, start=start, time_limit=time_limit)
            if found is not None:
                break
        
        if found is None:
            break
            
        # Remove used edges
        for i in range(len(found)):
            u = found[i]
            v = found[(i+1) % len(found)]
            available.discard(edge_key(u, v))
        cycles.append(found)
        
        # Check if remaining graph can still support a Hamiltonian cycle
        if not min_degree(available) or not is_connected(available):
            break

    if cycles:
        result = [list(c) for c in cycles]
        return result
    else:
        # Fallback: try to find at least 1 cycle
        random.seed(42)
        for _ in range(50):
            found = find_hamiltonian_cycle(set(edge_set), start=random.randint(0, n-1), time_limit=2.0)
            if found:
                return [found]
        return [list(range(n))]
# EVOLVE-BLOCK-END
