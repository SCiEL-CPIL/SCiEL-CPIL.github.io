import time
import sys

sys.setrecursionlimit(10000)

P = {
    'n': 21, 
    'edges': [
        [0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], 
        [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], 
        [2, 3], [2, 4], [2, 7], [2, 18], 
        [3, 4], [3, 5], [3, 8], [3, 19], 
        [4, 5], [4, 6], [4, 9], [4, 20], 
        [5, 6], [5, 7], [5, 10], 
        [6, 7], [6, 8], [6, 11], 
        [7, 8], [7, 9], [7, 12], 
        [8, 9], [8, 10], [8, 13], 
        [9, 10], [9, 11], [9, 14], 
        [10, 11], [10, 12], [10, 15], 
        [11, 12], [11, 13], [11, 16], 
        [12, 13], [12, 14], [12, 17], 
        [13, 14], [13, 15], [13, 18], 
        [14, 15], [14, 16], [14, 19], 
        [15, 16], [15, 17], [15, 20], 
        [16, 17], [16, 18], 
        [17, 18], [17, 19], 
        [18, 19], [18, 20], 
        [19, 20]
    ]
}

def construct():
    n = P['n']
    start_time = time.time()
    deadline = start_time + 50.0
    
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(P['edges']):
        adj[u].append((v, i))
        adj[v].append((u, i))
    
    def find_ham_cycle(used_mask, start_vertex):
        path = [start_vertex]
        visited_mask = 1 << start_vertex
        
        def dfs(current, path, visited_mask, used_mask):
            if time.time() > deadline:
                return None
            
            if len(path) == n:
                for nb, e_idx in adj[current]:
                    if nb == path[0]:
                        if not (used_mask & (1 << e_idx)):
                            return path[:]
                return None
            
            candidates = []
            for neighbor, e_idx in adj[current]:
                if visited_mask & (1 << neighbor):
                    continue
                if used_mask & (1 << e_idx):
                    continue
                avail = 0
                for nn, ee in adj[neighbor]:
                    if not (visited_mask & (1 << nn)) and not (used_mask & (1 << ee)):
                        avail += 1
                candidates.append((avail, neighbor, e_idx))
            
            if not candidates:
                return None
            
            candidates.sort(key=lambda x: x[0])
            
            for avail, neighbor, e_idx in candidates:
                new_used = used_mask | (1 << e_idx)
                new_visited = visited_mask | (1 << neighbor)
                path.append(neighbor)
                
                res = dfs(neighbor, path, new_visited, new_used)
                if res is not None:
                    return res
                
                path.pop()
            
            return None
        
        return dfs(start_vertex, path, visited_mask, used_mask)
    
    cycles = []
    used_mask = 0
    
    for _ in range(3):
        if time.time() > deadline:
            break
        
        found = False
        for start_v in range(n):
            if time.time() > deadline:
                break
            
            cycle = find_ham_cycle(used_mask, start_v)
            if cycle is not None:
                cycles.append(cycle)
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    for nb, e_idx in adj[u]:
                        if nb == v:
                            used_mask |= (1 << e_idx)
                            break
                found = True
                break
        
        if not found:
            break
    
    if not cycles:
        return []
    
    vertex_set = set(cycles[0])
    for c in cycles:
        if set(c) != vertex_set:
            return []
    
    edge_set = set()
    for c in cycles:
        for i in range(len(c)):
            u = c[i]
            v = c[(i + 1) % len(c)]
            edge = (min(u, v), max(u, v))
            if edge in edge_set:
                return []
            edge_set.add(edge)
    
    return cycles
