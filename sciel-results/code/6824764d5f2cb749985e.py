import sys
import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    adj = defaultdict(list)
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    
    used_edges = set()
    
    def is_edge_available(u, v):
        if u > v:
            u, v = v, u
        return (u, v) not in used_edges

    def find_ham_cycle(start):
        path = [start]
        visited = {start}
        
        def dfs(current, path, visited):
            if len(path) == n:
                if is_edge_available(current, start):
                    return list(path)
                return None
            
            neighbors = []
            for neighbor in adj[current]:
                if neighbor not in visited and is_edge_available(current, neighbor):
                    neighbors.append(neighbor)
            
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                visited.add(neighbor)
                path.append(neighbor)
                
                result = dfs(neighbor, path, visited)
                if result is not None:
                    return result
                    
                path.pop()
                visited.remove(neighbor)
            
            return None

        return dfs(start, path, visited)

    best_cycles = []
    
    for attempt in range(20):
        used_edges = set()
        cycles = []
        success = True
        for _ in range(3):
            start = random.randint(0, n-1)
            cycle = find_ham_cycle(start)
            if cycle is None:
                success = False
                break
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        
        if success and len(cycles) > len(best_cycles):
            best_cycles = cycles
            if len(best_cycles) == 3:
                break
        
    if not best_cycles:
        used_edges = set()
        cycles = []
        for _ in range(3):
            start = random.randint(0, n-1)
            cycle = find_ham_cycle(start)
            if cycle is None:
                break
            cycles.append(cycle)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if u > v:
                    u, v = v, u
                used_edges.add((u, v))
        best_cycles = cycles

    return best_cycles

# EVOLVE-BLOCK-END
