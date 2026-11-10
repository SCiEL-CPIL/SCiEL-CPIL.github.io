import copy
import time

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    # Build adjacency list as list of sets
    adj = [set() for _ in range(n)]
    for u, v in P['edges']:
        adj[u].add(v)
        adj[v].add(u)
    
    def find_ham_cycle(adj_in):
        # adj_in is a list of sets
        # We will mutate a copy
        adj_work = [set(s) for s in adj_in]
        
        # Warnsdorff's heuristic: sort neighbors by their degree in adj_work
        def get_neighbors(v):
            neighbors = list(adj_work[v])
            # Sort by number of available neighbors (degree in remaining graph)
            neighbors.sort(key=lambda x: len(adj_work[x]))
            return neighbors

        def dfs(path):
            if len(path) == n:
                # Check if last vertex connects to first
                if path[0] in adj_work[path[-1]]:
                    return path[:]
                return None
            
            last = path[-1]
            # Get neighbors of last, sorted
            for u in get_neighbors(last):
                if u in path:
                    continue
                
                # Remove edges last-u
                adj_work[last].remove(u)
                adj_work[u].remove(last)
                
                res = dfs(path + [u])
                if res is not None:
                    return res
                
                # Restore edges
                adj_work[last].add(u)
                adj_work[u].add(last)
                
            return None

        # Start from vertex 0
        return dfs([0])

    # Find first cycle
    cycle1 = find_ham_cycle(adj)
    if cycle1 is None:
        return []
        
    # Remove edges of cycle1 from adj
    for i in range(len(cycle1)):
        u = cycle1[i]
        v = cycle1[(i+1) % len(cycle1)]
        adj[u].remove(v)
        adj[v].remove(u)
        
    # Find second cycle
    cycle2 = find_ham_cycle(adj)
    
    if cycle2 is not None:
        return [cycle1, cycle2]
    else:
        return [cycle1]
