"""Packing edge-disjoint Hamiltonian cycles on 21-vertex host graph."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def _edge_key(u, v):
    return (min(u, v), max(u, v))

def _build_remaining_adj(edges, n, used_edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        ek = _edge_key(u, v)
        if ek not in used_edges:
            adj[u].add(v)
            adj[v].add(u)
    return adj

def _find_hamiltonian_cycle(adj, n, deadline, rng):
    """Find a Hamiltonian cycle using Warnsdorff-guided randomized backtracking."""
    
    def dfs(v, visited, path, start):
        if time.time() > deadline:
            return None
        if len(path) == n:
            # Check if we can close the cycle
            if start in adj[v]:
                return path[:]
            return None
        
        # Warnsdorff: order neighbors by fewest unvisited neighbors
        unvisited_neighbors = [u for u in adj[v] if u not in visited]
        
        if not unvisited_neighbors:
            return None
        
        # If this is the last step before closing, must go to start
        if len(path) == n - 1:
            if start in unvisited_neighbors:
                unvisited_neighbors = [start]
            else:
                return None
        
        # Score by number of unvisited neighbors (Warnsdorff's rule)
        def score(u):
            if u == start:
                return 0  # prioritize closing
            cnt = sum(1 for w in adj[u] if w not in visited and w != start)
            return cnt
        
        unvisited_neighbors.sort(key=score)
        
        # Randomize order slightly for diversity
        if len(unvisited_neighbors) > 1:
            for i in range(len(unvisited_neighbors) - 1, 0, -1):
                j = rng.randint(0, i)
                unvisited_neighbors[i], unvisited_neighbors[j] = unvisited_neighbors[j], unvisited_neighbors[i]
        
        for u in unvisited_neighbors:
            visited.add(u)
            path.append(u)
            result = dfs(u, visited, path, start)
            if result is not None:
                return result
            path.pop()
            visited.remove(u)
            if time.time() > deadline:
                return None
        return None
    
    # Try different starting vertices
    starts = list(range(n))
    rng.shuffle(starts)
    
    for start in starts:
        if time.time() > deadline:
            break
        visited = {start}
        path = [start]
        result = dfs(start, visited, path, start)
        if result is not None:
            return result
    
    return None

def construct():
    n = P['n']
    edges = P['edges']
    rng = random.Random(12345)
    
    deadline = time.time() + 84.0  # 6s safety margin
    
    used_edges = set()
    cycles = []
    
    # Try to extract up to 3 edge-disjoint Hamiltonian cycles
    for _ in range(3):
        if time.time() > deadline:
            break
        
        remaining_adj = _build_remaining_adj(edges, n, used_edges)
        
        # Check minimum degree
        if any(len(remaining_adj[i]) < 2 for i in range(n)):
            break
        
        # Try multiple restarts to find a Hamiltonian cycle
        found = None
        for attempt in range(100):
            if time.time() > deadline:
                break
            found = _find_hamiltonian_cycle(remaining_adj, n, deadline, rng)
            if found:
                break
        
        if not found:
            break
        
        # Validate and mark edges
        valid = True
        edge_set = set()
        for i in range(n):
            u = found[i]
            v = found[(i + 1) % n]
            ek = _edge_key(u, v)
            if ek in used_edges or ek in edge_set:
                valid = False
                break
            edge_set.add(ek)
        
        if not valid or len(set(found)) != n:
            break
        
        used_edges.update(edge_set)
        cycles.append(list(found))
    
    if cycles:
        return [list(c) for c in cycles]
    else:
        return [list(range(n))]
# EVOLVE-BLOCK-END
