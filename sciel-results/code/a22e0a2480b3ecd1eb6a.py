import time
import sys
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    all_edges = set()
    for u, v in edges:
        if u > v:
            u, v = v, u
        all_edges.add((u, v))
    
    # Adjacency list for quick neighbor lookup
    adj = {i: set() for i in range(n)}
    for u, v in all_edges:
        adj[u].add(v)
        adj[v].add(u)
        
    remaining_edges = set(all_edges)
    cycles = []
    start_time = time.time()
    deadline = start_time + 85  # 85s limit to be safe within 90s
    
    def find_hamiltonian_cycle(rem_edges):
        # Pruning: check if any vertex has degree < 2 in rem_edges
        deg = {i: 0 for i in range(n)}
        for u, v in rem_edges:
            deg[u] += 1
            deg[v] += 1
        for i in range(n):
            if deg[i] < 2:
                return None
            
        # Backtracking with better heuristics
        # We fix the start vertex to 0 to reduce symmetry
        # path[0] = 0
        # We need to visit all n vertices and return to 0.
        
        visited = [False] * n
        visited[0] = True
        path = [0]
        
        def backtrack():
            if time.time() > deadline:
                return False
            if len(path) == n:
                # Check if last vertex is connected to 0
                last = path[-1]
                e_key = (min(0, last), max(0, last))
                if e_key in rem_edges:
                    return True
                return False
            
            curr = path[-1]
            # Get neighbors of curr that are in rem_edges and not visited
            neighbors = []
            for v in adj[curr]:
                if not visited[v]:
                    e_key = (min(curr, v), max(curr, v))
                    if e_key in rem_edges:
                        neighbors.append(v)
            
            if not neighbors:
                return False
            
            # Heuristic: try vertices with fewer remaining options first (most constrained)
            # Sort by degree in remaining graph (ascending)
            neighbors.sort(key=lambda v: deg[v])
            
            for v in neighbors:
                # Additional pruning: check if removing this edge would leave any vertex with degree < 2
                # (except for the current vertex and the next vertex which are being connected)
                visited[v] = True
                path.append(v)
                
                if backtrack():
                    return True
                
                path.pop()
                visited[v] = False
            
            return False
            
        if backtrack():
            return list(path)
        return None

    # Try to find up to 3 edge-disjoint Hamiltonian cycles
    # The graph is 6-regular, so theoretically we can have at most 3 edge-disjoint Hamiltonian cycles
    while len(cycles) < 3:
        if time.time() > deadline:
            break
        cycle = find_hamiltonian_cycle(remaining_edges)
        if cycle is None:
            break
        cycles.append(cycle)
        # Remove edges from remaining
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e_key = (min(u, v), max(u, v))
            if e_key in remaining_edges:
                remaining_edges.remove(e_key)
            else:
                break
        
    # Validate and format output
    if not cycles:
        # Fallback: return a single Hamiltonian cycle if we found none (should not happen)
        return [list(range(n))]
        
    # Convert to list of lists of ints
    result = []
    for c in cycles:
        result.append([int(v) for v in c])
        
    return result

# EVOLVE-BLOCK-END
