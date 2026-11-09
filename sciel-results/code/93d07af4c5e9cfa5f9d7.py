"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    n = P['n']
    all_edges = set()
    adj = {i: set() for i in range(n)}
    for u, v in P['edges']:
        if u > v:
            u, v = v, u
        all_edges.add((u, v))
        adj[u].add(v)
        adj[v].add(u)
        
    available_edges = set(all_edges)
    cycles = []
    
    def find_ham_cycle():
        path = [0]
        visited = {0}
        
        def backtrack():
            if time.time() > deadline:
                return False
                
            if len(path) == n:
                u, v = path[-1], path[0]
                e = (u, v) if u < v else (v, u)
                if e in available_edges:
                    return True
                return False
            
            u = path[-1]
            # Sort neighbors for determinism
            for v in sorted(adj[u]):
                if v in visited:
                    continue
                e = (u, v) if u < v else (v, u)
                if e not in available_edges:
                    continue
                
                # Simple pruning: if v is not the last vertex to be visited,
                # it must have at least one other unvisited neighbor or be connected to start?
                # Actually, standard backtracking is fine for N=21.
                
                path.append(v)
                visited.add(v)
                
                if backtrack():
                    return True
                
                path.pop()
                visited.remove(v)
                
            return False
        
        if backtrack():
            return path
        return None

    while len(cycles) < 10:
        if time.time() > deadline:
            break
            
        cycle = find_ham_cycle()
        if cycle is None:
            break
            
        cycles.append(cycle)
        
        # Remove edges from available
        for i in range(len(cycle)):
            u = cycle[i]
            v = cycle[(i + 1) % len(cycle)]
            e = (u, v) if u < v else (v, u)
            if e in available_edges:
                available_edges.remove(e)

    # Ensure return type is list of lists of ints
    return [list(c) for c in cycles]
# EVOLVE-BLOCK-END
