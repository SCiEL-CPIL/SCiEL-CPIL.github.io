"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    edges = P['edges']
    m = len(edges)
    
    # Build edge index and adjacency
    edge_set = set()
    edge_index = {}
    for i, (u, v) in enumerate(edges):
        key = (min(u, v), max(u, v))
        edge_set.add(key)
        edge_index[key] = i
    
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    deadline = time.time() + 80
    
    # Strategy: Use a constraint satisfaction approach
    # Each vertex must have exactly 2 edges in each cycle
    # We'll try to find 3 cycles simultaneously using backtracking
    
    def find_three_cycles():
        # Try multiple random orderings of edges
        for attempt in range(50):
            if time.time() > deadline - 2:
                break
            
            # Randomize edge order
            edge_order = list(range(m))
            random.shuffle(edge_order)
            
            # assignment[i] = which cycle (0,1,2) edge i is assigned to, or -1 if unassigned
            assignment = [-1] * m
            
            # For each vertex and each cycle, count how many edges are assigned
            # vertex_cycle_count[v][c] = number of edges of cycle c incident to v
            vcc = [[0, 0, 0] for _ in range(n)]
            
            # Track which vertices have completed their degree-2 in each cycle
            # A vertex is "done" in cycle c if vcc[v][c] == 2
            
            # We need to ensure connectivity - each cycle must be a single cycle
            # This is hard to check incrementally, so we'll use a heuristic approach
            
            # Alternative: Find one cycle at a time with strong pruning
            # But ensure the remaining graph can support 2 more cycles
            
            # Let's use a different approach: 
            # 1. Find first Hamiltonian cycle
            # 2. Remove its edges
            # 3. The remaining graph is 4-regular
            # 4. Find a Hamiltonian cycle in the 4-regular graph
            # 5. Remove its edges
            # 6. The remaining graph must be a 2-regular graph = union of cycles
            # 7. Check if it's a single Hamiltonian cycle
            
            # For step 1, use backtracking with Warnsdorff's rule
            
            def find_hc(adj_list, start=0):
                path = [start]
                visited = {start}
                
                def dfs(u):
                    if time.time() > deadline - 2:
                        return None
                    
                    if len(path) == n:
                        # Check if closing edge exists
                        if start in adj_list[u]:
                            return list(path)
                        return None
                    
                    # Get unvisited neighbors
                    unvisited_neighbors = [v for v in adj_list[u] if v not in visited]
                    
                    if not unvisited_neighbors:
                        return None
                    
                    # Warnsdorff: prioritize neighbors with fewer unvisited neighbors
                    scored = []
                    for v in unvisited_neighbors:
                        # Count unvisited neighbors of v (excluding u which is now visited)
                        count = sum(1 for w in adj_list[v] if w not in visited and w != u)
                        scored.append((count, v))
                    
                    scored.sort()
                    
                    for _, v in scored:
                        path.append(v)
                        visited.add(v)
                        
                        res = dfs(v)
                        if res is not None:
                            return res
                        
                        path.pop()
                        visited.remove(v)
                    
                    return None
                
                return dfs(start)
            
            # Find first cycle
            adj_cur = defaultdict(list)
            for u, v in edges:
                adj_cur[u].append(v)
                adj_cur[v].append(u)
            
            hc1 = find_hc(adj_cur)
            if hc1 is None:
                continue
            
            # Remove hc1 edges
            used_edges = set()
            for i in range(len(hc1)):
                u = hc1[i]
                v = hc1[(i + 1) % len(hc1)]
                used_edges.add((min(u, v), max(u, v)))
            
            # Build remaining graph
            adj_rem = defaultdict(list)
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in used_edges:
                    adj_rem[u].append(v)
                    adj_rem[v].append(u)
            
            # Check all vertices have degree 4
            possible = True
            for i in range(n):
                if len(adj_rem[i]) != 4:
                    possible = False
                    break
            if not possible:
                continue
            
            # Find second cycle in remaining 4-regular graph
            hc2 = find_hc(adj_rem)
            if hc2 is None:
                continue
            
            # Remove hc2 edges
            for i in range(len(hc2)):
                u = hc2[i]
                v = hc2[(i + 1) % len(hc2)]
                used_edges.add((min(u, v), max(u, v)))
            
            # Build final remaining graph
            adj_final = defaultdict(list)
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in used_edges:
                    adj_final[u].append(v)
                    adj_final[v].append(u)
            
            # Check all vertices have degree 2
            possible = True
            for i in range(n):
                if len(adj_final[i]) != 2:
                    possible = False
                    break
            if not possible:
                continue
            
            # The remaining graph is 2-regular, so it's a union of cycles
            # Check if it's a single Hamiltonian cycle
            # Start from vertex 0 and follow the cycle
            visited_final = set()
            cycle = [0]
            visited_final.add(0)
            current = 0
            
            while True:
                # Find next vertex in cycle
                nexts = [v for v in adj_final[current] if v not in visited_final]
                if not nexts:
                    # Check if we can close the cycle
                    if 0 in adj_final[current] and len(cycle) == n:
                        break
                    else:
                        # Not a Hamiltonian cycle
                        break
                current = nexts[0]
                cycle.append(current)
                visited_final.add(current)
                if len(cycle) > n:
                    break
            
            if len(cycle) == n and 0 in adj_final[cycle[-1]]:
                # Verify edge disjointness
                all_edges_used = set()
                for c in [hc1, hc2, cycle]:
                    for i in range(len(c)):
                        u = c[i]
                        v = c[(i + 1) % len(c)]
                        e = (min(u, v), max(u, v))
                        if e in all_edges_used:
                            break
                        all_edges_used.add(e)
                    else:
                        continue
                    break
                else:
                    return [list(hc1), list(hc2), list(cycle)]
        
        return None
    
    result = find_three_cycles()
    
    if result is not None:
        # Convert to proper types
        return [[int(v) for v in c] for c in result]
    
    # Fallback: try to find at least 2 cycles
    def find_two_cycles():
        for attempt in range(100):
            if time.time() > deadline - 2:
                break
            
            adj_cur = defaultdict(list)
            for u, v in edges:
                adj_cur[u].append(v)
                adj_cur[v].append(u)
            
            def find_hc(adj_list, start=0):
                path = [start]
                visited = {start}
                
                def dfs(u):
                    if time.time() > deadline - 2:
                        return None
                    
                    if len(path) == n:
                        if start in adj_list[u]:
                            return list(path)
                        return None
                    
                    unvisited_neighbors = [v for v in adj_list[u] if v not in visited]
                    if not unvisited_neighbors:
                        return None
                    
                    scored = []
                    for v in unvisited_neighbors:
                        count = sum(1 for w in adj_list[v] if w not in visited and w != u)
                        scored.append((count, v))
                    
                    scored.sort()
                    
                    for _, v in scored:
                        path.append(v)
                        visited.add(v)
                        
                        res = dfs(v)
                        if res is not None:
                            return res
                        
                        path.pop()
                        visited.remove(v)
                    
                    return None
                
                return dfs(start)
            
            hc1 = find_hc(adj_cur)
            if hc1 is None:
                continue
            
            used_edges = set()
            for i in range(len(hc1)):
                u = hc1[i]
                v = hc1[(i + 1) % len(hc1)]
                used_edges.add((min(u, v), max(u, v)))
            
            adj_rem = defaultdict(list)
            for u, v in edges:
                key = (min(u, v), max(u, v))
                if key not in used_edges:
                    adj_rem[u].append(v)
                    adj_rem[v].append(u)
            
            hc2 = find_hc(adj_rem)
            if hc2 is None:
                continue
            
            return [list(hc1), list(hc2)]
        
        return None
    
    result = find_two_cycles()
    if result is not None:
        return [[int(v) for v in c] for c in result]
    
    # Fallback: find one cycle
    adj_cur = defaultdict(list)
    for u, v in edges:
        adj_cur[u].append(v)
        adj_cur[v].append(u)
    
    def find_hc(adj_list, start=0):
        path = [start]
        visited = {start}
        
        def dfs(u):
            if time.time() > deadline - 1:
                return None
            
            if len(path) == n:
                if start in adj_list[u]:
                    return list(path)
                return None
            
            unvisited_neighbors = [v for v in adj_list[u] if v not in visited]
            if not unvisited_neighbors:
                return None
            
            scored = []
            for v in unvisited_neighbors:
                count = sum(1 for w in adj_list[v] if w not in visited and w != u)
                scored.append((count, v))
            
            scored.sort()
            
            for _, v in scored:
                path.append(v)
                visited.add(v)
                
                res = dfs(v)
                if res is not None:
                    return res
                
                path.pop()
                visited.remove(v)
            
            return None
        
        return dfs(start)
    
    hc = find_hc(adj_cur)
    if hc is not None:
        return [[int(v) for v in hc]]
    
    # Last resort: return a simple triangle if it exists
    for u, v in edges:
        for w in adj[u]:
            if w != v and (v, w) in edge_set or (w, v) in edge_set:
                if u < v < w:
                    return [[int(u), int(v), int(w)]]
    
    # Should not reach here
    return [[0, 1, 2]]

# EVOLVE-BLOCK-END
