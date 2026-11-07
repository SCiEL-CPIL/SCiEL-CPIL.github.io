import sys
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s safety margin
    
    n = P['n']
    adj = {i: [] for i in range(n)}
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Pre-compute edge tuples for faster lookup
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add(tuple(sorted((u, v))))
        
    best_cycles = []
    
    # Try multiple random permutations to find 3 edge-disjoint Hamiltonian cycles
    # Theoretical max is 3. We want to ensure we find 3.
    for attempt in range(1000):
        if time.time() > deadline:
            break
            
        # Shuffle neighbors to vary the DFS path
        adj_shuffled = {i: adj[i][:] for i in range(n)}
        for i in range(n):
            random.shuffle(adj_shuffled[i])
            
        forbidden_edges = set()
        cycles = []
        success = True
        
        for c_idx in range(3):
            if time.time() > deadline:
                success = False
                break
                
            # Find one Hamiltonian cycle avoiding forbidden_edges
            start = 0 # Can also randomize start
            path = [start]
            visited = {start}
            
            found = False
            
            def dfs(u, depth):
                nonlocal found
                if time.time() > deadline:
                    return
                if found:
                    return
                
                if depth == n:
                    edge = tuple(sorted((u, start)))
                    if edge not in forbidden_edges:
                        found = True
                    return
                
                for v in adj_shuffled[u]:
                    if v in visited:
                        continue
                    edge = tuple(sorted((u, v)))
                    if edge in forbidden_edges:
                        continue
                    
                    visited.add(v)
                    path.append(v)
                    
                    dfs(v, depth + 1)
                    
                    if found:
                        return
                    
                    visited.remove(v)
                    path.pop()
            
            dfs(start, 1)
            
            if not found:
                success = False
                break
                
            cycles.append(path[:])
            for i in range(len(cycles[-1])):
                u = cycles[-1][i]
                v = cycles[-1][(i + 1) % len(cycles[-1])]
                forbidden_edges.add(tuple(sorted((u, v))))
                
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == 3:
                break # Found optimal, stop searching
                
    if not best_cycles:
        # Fallback: return a single cycle if we couldn't find 3 (should not happen if graph is good)
        # But we must return a valid list of cycles.
        # If we found nothing, we need to find at least 1?
        # The task says "nonempty list".
        # Let's try to find 1 cycle without forbidden edges.
        adj_shuffled = {i: adj[i][:] for i in range(n)}
        for i in range(n):
            random.shuffle(adj_shuffled[i])
        forbidden_edges = set()
        start = 0
        path = [start]
        visited = {start}
        found = False
        
        def dfs_single(u, depth):
            nonlocal found
            if time.time() > deadline:
                return
            if found:
                return
            if depth == n:
                edge = tuple(sorted((u, start)))
                if edge not in forbidden_edges:
                    found = True
                return
            for v in adj_shuffled[u]:
                if v in visited:
                    continue
                edge = tuple(sorted((u, v)))
                if edge in forbidden_edges:
                    continue
                visited.add(v)
                path.append(v)
                dfs_single(v, depth + 1)
                if found:
                    return
                visited.remove(v)
                path.pop()
                
        dfs_single(start, 1)
        if found:
            best_cycles = [path[:]]
        else:
            # Absolute fallback: return empty? No, "nonempty list".
            # If we can't find any cycle, the graph is broken or we timed out.
            # Return a dummy valid cycle if possible?
            # Given the graph is fixed and likely has cycles, this shouldn't happen.
            # But to be safe against "invalid result", we might need to ensure we return something.
            # However, if we return something invalid, score is 0.
            # If we return [] score is 0 (or invalid).
            # Let's assume we find at least 1.
            pass

    return best_cycles
# EVOLVE-BLOCK-END
