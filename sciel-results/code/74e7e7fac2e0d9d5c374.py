import time
import random
import heapq
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    best_n = 39
    best_edges = []
    
    for target_n in range(47, 39, -1):
        if time.time() > deadline:
            break
            
        min_deg_needed = target_n - 39
        
        found = False
        # Increase restarts to explore more random orderings
        for restart in range(500):
            if time.time() > deadline:
                break
                
            random.seed(restart * 12345 + 789)
            
            # Use a 2D array for faster membership checking
            in_graph = [[False] * target_n for _ in range(target_n)]
            adj = [0] * target_n
            deg = [0] * target_n
            edges = []
            
            def can_add(u, v):
                nu = adj[u]
                nv = adj[v]
                c = nu
                while c:
                    a_bit = c & -c
                    a = a_bit.bit_length() - 1
                    c ^= a_bit
                    if (nv & adj[a]) != 0:
                        return False
                return True

            heap = []
            
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    heapq.heappush(heap, (0, u, v))
            
            while heap:
                if time.time() > deadline:
                    break
                    
                pri, u, v = heapq.heappop(heap)
                
                if in_graph[u][v]:
                    continue
                
                if pri != deg[u] + deg[v]:
                    continue
                
                if can_add(u, v):
                    adj[u] |= 1 << v
                    adj[v] |= 1 << u
                    deg[u] += 1
                    deg[v] += 1
                    edges.append([u, v])
                    in_graph[u][v] = True
                    in_graph[v][u] = True
                    
                    for w in range(target_n):
                        if w == u or w == v:
                            continue
                        if not in_graph[u][w]:
                            heapq.heappush(heap, (deg[u] + deg[w], u, w))
                        if not in_graph[v][w]:
                            heapq.heappush(heap, (deg[v] + deg[w], v, w))
                    
                else:
                    pass

            if min(deg) >= min_deg_needed:
                if target_n > best_n:
                    best_n = target_n
                    best_edges = edges
                found = True
                break
            
        if best_n >= target_n:
            break
            
    if best_n >= 40:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
