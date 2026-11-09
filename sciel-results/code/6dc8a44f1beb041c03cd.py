import time
import random

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 50s safety margin
    
    # Initial graph
    edges = set()
    for e in P['edges']:
        edges.add((min(e[0], e[1]), max(e[0], e[1])))
    
    n = P['n']
    cycles_found = []
    current_edges = edges.copy()
    
    # We want to find up to 3 cycles
    max_cycles = 3
    
    # Helper to find a Hamiltonian cycle in the current graph
    def find_ham_cycle(edge_set, n, time_limit):
        # Build adjacency
        adj = {i: set() for i in range(n)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
            
        # Check connectivity quickly? 
        # For now, just DFS.
        
        # Sort neighbors by degree (Warnsdorff)
        # This is static, might need dynamic sorting in DFS for better performance
        # But static is faster. Let's try static first.
        # Actually, dynamic sorting in DFS is better for Hamiltonian path.
        
        visited = [False] * n
        path = [0]
        visited[0] = True
        
        def dfs(v, depth):
            if time.time() > time_limit:
                return None
            
            if depth == n:
                if 0 in adj[v]:
                    return path[:]
                return None
            
            # Get neighbors
            neighbors = list(adj[v])
            # Sort by number of unvisited neighbors
            def sort_key(u):
                if u == 0:
                    return 100 # Avoid 0 until the end? 
                    # Actually, if we are not at depth n-1, we can't go to 0.
                    # So 0 should be last.
                cnt = 0
                for w in adj[u]:
                    if not visited[w]:
                        cnt += 1
                return cnt
            
            neighbors.sort(key=sort_key)
            
            for u in neighbors:
                if not visited[u]:
                    visited[u] = True
                    path.append(u)
                    
                    res = dfs(u, depth + 1)
                    if res:
                        return res
                        
                    path.pop()
                    visited[u] = False
            return None
        
        # Try starting from 0
        # Randomize if needed?
        # For now, deterministic. If it fails, we might need restarts.
        # But finding 1 ham cycle is usually fast.
        # The difficulty is finding the 2nd and 3rd after removing edges.
        
        return dfs(0, 1)

    # We might need multiple attempts if the first choice of C1 makes C2 impossible.
    # So we should use a backtracking search over the choices of C1, C2, C3.
    # Or, simpler: Randomized restarts for the whole process.
    
    best_cycles = []
    
    for attempt in range(1000): # Max attempts
        if time.time() > deadline:
            break
            
        current_edges = edges.copy()
        cycles = []
        success = True
        
        for i in range(max_cycles):
            # Shuffle edges slightly? No, just find a cycle.
            # To diversify, we can randomize the neighbor order in the DFS.
            # Let's modify find_ham_cycle to accept a random seed or use global random.
            
            # Re-implementing DFS with randomization for robustness
            adj = {i: set() for i in range(n)}
            for u, v in current_edges:
                adj[u].add(v)
                adj[v].add(u)
                
            visited = [False] * n
            path = [0]
            visited[0] = True
            
            def dfs_rand(v, depth):
                if time.time() > deadline:
                    return None
                
                if depth == n:
                    if 0 in adj[v]:
                        return path[:]
                    return None
                
                neighbors = [u for u in adj[v] if not visited[u]]
                # Sort by degree heuristic, but add randomness
                # Or just shuffle?
                # Warnsdorff is strong. Let's stick to it but maybe randomize ties.
                def sort_key(u):
                    cnt = 0
                    for w in adj[u]:
                        if not visited[w]:
                            cnt += 1
                    return (cnt, random.random())
                
                neighbors.sort(key=sort_key)
                
                for u in neighbors:
                    visited[u] = True
                    path.append(u)
                    
                    res = dfs_rand(u, depth + 1)
                    if res:
                        return res
                        
                    path.pop()
                    visited[u] = False
                return None
            
            c = dfs_rand(0, 1)
            
            if c is None:
                success = False
                break
            else:
                cycles.append(c)
                # Remove edges of c from current_edges
                for i in range(len(c)):
                    u = c[i]
                    v = c[(i+1) % len(c)]
                    e = (min(u, v), max(u, v))
                    if e in current_edges:
                        current_edges.remove(e)
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == max_cycles:
                break # Found optimal (max possible given edge count)
                
    # If we didn't find 3, return what we have.
    # But we know 3 is possible.
    # If best_cycles is empty, return the seed result?
    # No, return whatever we found.
    
    # Convert to required format: list of lists of ints
    result = []
    for c in best_cycles:
        result.append(list(c))
        
    return result
