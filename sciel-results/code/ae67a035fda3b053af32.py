import time
import random
from itertools import permutations

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    deadline = time.time() + 50.0  # safety margin under 60s
    n = P['n']
    all_vertices = list(range(n))
    
    # Build adjacency with edge set
    edge_set = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        edge_set.add((min(u,v), max(u,v)))
        adj[u].add(v)
        adj[v].add(u)
    
    def edge_key(u, v):
        return (min(u,v), max(u,v))
    
    def find_hamiltonian_cycle(available_edges, start=0, time_limit=5.0):
        """Find a Hamiltonian cycle using only edges in available_edges.
        Returns list of vertices (cycle) or None."""
        t0 = time.time()
        vertices = list(range(n))
        # Fix start vertex to break symmetry
        start = 0
        path = [start]
        visited = {start}
        
        def backtrack():
            if time.time() - t0 > time_limit:
                return None
            if len(path) == n:
                # Check closing edge
                last = path[-1]
                if edge_key(last, start) in available_edges:
                    return list(path)
                return None
            last = path[-1]
            # Try neighbors
            neighbors = list(adj[last])
            random.shuffle(neighbors)
            for nb in neighbors:
                if nb not in visited:
                    if edge_key(last, nb) in available_edges:
                        path.append(nb)
                        visited.add(nb)
                        result = backtrack()
                        if result is not None:
                            return result
                        path.pop()
                        visited.remove(nb)
            return None
        
        result = backtrack()
        return result
    
    # Greedily find edge-disjoint Hamiltonian cycles on full vertex set
    cycles = []
    available = set(edge_set)
    random.seed(42)
    
    max_attempts = 10
    for attempt in range(max_attempts):
        if time.time() > deadline:
            break
        # Try multiple random restarts for this cycle
        found = None
        for restart in range(50):
            if time.time() > deadline:
                break
            random.shuffle(list(adj[0]))  # shuffle adjacency order
            found = find_hamiltonian_cycle(available, start=0, time_limit=2.0)
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
    
    if cycles:
        # Return as list of lists of ints
        result = [list(c) for c in cycles]
        return result
    else:
        # Fallback: return seed
        return [list(range(n))]
# EVOLVE-BLOCK-END
