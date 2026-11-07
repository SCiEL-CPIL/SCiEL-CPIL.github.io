import sys
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin (90s limit, leave 35s buffer)
    
    n = P['n']
    adj = {i: [] for i in range(n)}
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    # Pre-compute edge tuples for faster lookup
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
        
    best_cycles = []
    
    # Try multiple random permutations to find 3 edge-disjoint Hamiltonian cycles
    # Theoretical max is 3. We want to ensure we find 3.
    for attempt in range(5000):
        if time.time() > deadline:
            break
            
        # Use Warnsdorff's heuristic: at each step, prefer neighbors with
        # fewest onward moves (minimum available unvisited neighbors)
        forbidden_edges = set()
        cycles = []
        success = True
        
        for c_idx in range(3):
            if time.time() > deadline:
                success = False
                break
                
            # Find one Hamiltonian cycle avoiding forbidden_edges
            # Try different start vertices to increase success rate
            start = c_idx  # Vary start vertex per cycle
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
                    edge = (min(u, start), max(u, start))
                    if edge not in forbidden_edges:
                        found = True
                    return
                
                # Warnsdorff heuristic: sort neighbors by number of onward moves
                candidates = []
                for v in adj[u]:
                    if v in visited:
                        continue
                    edge = (min(u, v), max(u, v))
                    if edge in forbidden_edges:
                        continue
                    
                    # Count onward moves from v (excluding u and visited nodes)
                    onward = 0
                    for w in adj[v]:
                        if w == u or w in visited:
                            continue
                        if w == start:
                            continue  # Can't go to start yet
                        e2 = (min(v, w), max(v, w))
                        if e2 not in forbidden_edges:
                            onward += 1
                    
                    candidates.append((onward, v))
                
                # Sort by onward moves (fewest first) - Warnsdorff
                candidates.sort(key=lambda x: x[0])
                
                for _, v in candidates:
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
                forbidden_edges.add((min(u, v), max(u, v)))
                
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == 3:
                break  # Found optimal, stop searching
                
    if not best_cycles:
        # Fallback: try with random start vertices
        for attempt in range(2000):
            if time.time() > deadline:
                break
                
            forbidden_edges = set()
            cycles = []
            success = True
            
            for c_idx in range(3):
                if time.time() > deadline:
                    success = False
                    break
                    
                start = random.randint(0, n - 1)
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
                        edge = (min(u, start), max(u, start))
                        if edge not in forbidden_edges:
                            found = True
                        return
                    
                    candidates = []
                    for v in adj[u]:
                        if v in visited:
                            continue
                        edge = (min(u, v), max(u, v))
                        if edge in forbidden_edges:
                            continue
                        
                        onward = 0
                        for w in adj[v]:
                            if w == u or w in visited:
                                continue
                            if w == start:
                                continue
                            e2 = (min(v, w), max(v, w))
                            if e2 not in forbidden_edges:
                                onward += 1
                        
                        candidates.append((onward, v))
                    
                    candidates.sort(key=lambda x: x[0])
                    
                    for _, v in candidates:
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
                    forbidden_edges.add((min(u, v), max(u, v)))
                    
            if success and len(cycles) > len(best_cycles):
                best_cycles = cycles
                if len(best_cycles) == 3:
                    break
    
    # Ensure all vertices are Python int
    result = []
    for cycle in best_cycles:
        result.append([int(v) for v in cycle])
    
    return result
# EVOLVE-BLOCK-END
