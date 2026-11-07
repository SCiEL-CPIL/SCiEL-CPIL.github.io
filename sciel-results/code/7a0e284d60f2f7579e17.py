"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    edge_set = set()
    adj = [[] for _ in range(n)]
    for e in edges:
        a, b = e
        edge_set.add((min(a, b), max(a, b)))
        adj[a].append(b)
        adj[b].append(a)
    
    def find_hamiltonian_cycle(used_edges):
        """Find a Hamiltonian cycle avoiding used edges using backtracking with Warnsdorff's rule."""
        import sys
        sys.setrecursionlimit(10000)
        
        def dfs(path, visited, current):
            if len(path) == n:
                # Check if current connects back to start
                start = path[0]
                if (min(current, start), max(current, start)) not in used_edges:
                    return path + [start]
                return None
            
            # Get unvisited neighbors not using used edges, sorted by degree (Warnsdorff)
            candidates = []
            for nb in adj[current]:
                if not visited[nb]:
                    edge = (min(current, nb), max(current, nb))
                    if edge not in used_edges:
                        # Count how many unvisited neighbors nb has (excluding used edges)
                        deg = 0
                        for nb2 in adj[nb]:
                            if not visited[nb2] and nb2 != current:
                                e2 = (min(nb, nb2), max(nb, nb2))
                                if e2 not in used_edges:
                                    deg += 1
                        candidates.append((deg, nb))
            
            candidates.sort()
            
            for deg, nb in candidates:
                edge = (min(current, nb), max(current, nb))
                if edge in used_edges:
                    continue
                visited[nb] = True
                result = dfs(path + [nb], visited, nb)
                if result is not None:
                    return result
                visited[nb] = False
            
            return None
        
        # Try starting from vertex 0
        for start in range(n):
            visited = [False] * n
            visited[start] = True
            result = dfs([start], visited, start)
            if result is not None:
                return result
        return None
    
    used_edges = set()
    cycles = []
    
    deadline = time.time() + 50  # Safety margin
    
    for i in range(10):
        if time.time() > deadline:
            break
        
        cycle = find_hamiltonian_cycle(used_edges)
        if cycle is None:
            break
        
        # Convert cycle to edge set and add to used_edges
        cycle_edges = set()
        for j in range(len(cycle) - 1):
            a, b = cycle[j], cycle[j+1]
            edge = (min(a, b), max(a, b))
            cycle_edges.add(edge)
        
        # Verify all edges are valid host edges
        valid = True
        for edge in cycle_edges:
            if edge not in edge_set:
                valid = False
                break
        
        if not valid:
            break
        
        used_edges.update(cycle_edges)
        cycles.append(cycle)
    
    # Ensure at least one cycle
    if not cycles:
        # Fallback: return a simple 3-cycle if possible
        # Find any triangle
        for i in range(n):
            for j in adj[i]:
                if j > i:
                    for k in adj[j]:
                        if k > j and k in adj[i]:
                            return [[i, j, k]]
        return [[0, 1, 2]]
    
    return cycles
# EVOLVE-BLOCK-END
