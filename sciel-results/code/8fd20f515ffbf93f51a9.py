import time
import random
from itertools import combinations

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = set()
    for u, v in P['edges']:
        base_edges.add((u, v))
        base_edges.add((v, u))
    
    # Candidate edges: Even to Odd
    candidates = []
    for u in range(n):
        for v in range(u+1, n):
            if (u % 2) != (v % 2) and (u, v) not in base_edges:
                candidates.append((u, v))
                
    def get_diameter(added_edges):
        # Build adjacency list
        adj = {i: [] for i in range(n)}
        for u, v in P['edges']:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
            
        # BFS from each node to find max distance
        max_d = 0
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = [start]
            while queue:
                u = queue.pop(0)
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        max_d = max(max_d, dist[v])
                        queue.append(v)
        return max_d

    def is_valid(added_edges):
        if get_diameter(added_edges) <= 4:
            return True
        return False

    best_solution = None
    start_time = time.time()
    
    # Greedy approach with restarts
    # We try to find a solution with minimum edges
    # We'll run greedy multiple times with slight variations
    
    for attempt in range(50):
        if time.time() - start_time > 45:
            break
            
        random.seed(attempt)
        current_edges = []
        
        # To add diversity, we can randomize the candidate order or tie-breaking
        cand_list = candidates[:]
        random.shuffle(cand_list)
        
        while True:
            d = get_diameter(current_edges)
            if d <= 4:
                break
            
            # Find the pair with max distance
            # We need to identify the bottleneck pair
            # For efficiency, we can just pick the pair that benefits most from a new edge
            # Let's find the actual max distance pair first
            adj = {i: [] for i in range(n)}
            for u, v in P['edges']:
                adj[u].append(v)
                adj[v].append(u)
            for u, v in current_edges:
                adj[u].append(v)
                adj[v].append(u)
                
            max_d = 0
            worst_pair = (0, 1)
            # Calculate all pairs distances to find worst
            # Since n=24, 24 BFS is fine
            for start in range(n):
                dist = [-1] * n
                dist[start] = 0
                queue = [start]
                while queue:
                    u = queue.pop(0)
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            if dist[v] > max_d:
                                max_d = dist[v]
                                worst_pair = (start, v)
                            queue.append(v)
            
            u, v = worst_pair
            
            best_edge = None
            best_score = float('inf')
            
            for edge in cand_list:
                if edge in current_edges:
                    continue
                x, y = edge
                # Estimate new distance between u and v
                # dist(u, x) + 1 + dist(y, v) or dist(u, y) + 1 + dist(x, v)
                # We need distances from u and v to x and y in the current graph
                # This requires BFS from u and v
                # To save time, we can approximate or just compute
                # Given n=24, computing BFS from u and v is cheap
                
                # BFS from u
                dist_u = [-1] * n
                dist_u[u] = 0
                q = [u]
                while q:
                    curr = q.pop(0)
                    for nb in adj[curr]:
                        if dist_u[nb] == -1:
                            dist_u[nb] = dist_u[curr] + 1
                            q.append(nb)
                            
                # BFS from v
                dist_v = [-1] * n
                dist_v[v] = 0
                q = [v]
                while q:
                    curr = q.pop(0)
                    for nb in adj[curr]:
                        if dist_v[nb] == -1:
                            dist_v[nb] = dist_v[curr] + 1
                            q.append(nb)
                
                d1 = dist_u[x] + 1 + dist_v[y]
                d2 = dist_u[y] + 1 + dist_v[x]
                new_d = min(d1, d2)
                
                # We want to minimize this new distance
                # Tie-breaking: maybe minimize sum of distances to other critical nodes?
                # For now, just minimize new_d
                if new_d < best_score:
                    best_score = new_d
                    best_edge = edge
            
            if best_edge is None:
                break
            
            current_edges.append(best_edge)
            
        if is_valid(current_edges):
            # Try to minimize by removing edges
            removed = True
            while removed:
                removed = False
                for i in range(len(current_edges)):
                    trial = current_edges[:i] + current_edges[i+1:]
                    if is_valid(trial):
                        current_edges = trial
                        removed = True
                        break
                        
            if best_solution is None or len(current_edges) < len(best_solution):
                best_solution = current_edges[:]
                
    if best_solution is None:
        # Fallback: return empty or a known valid large set
        # But greedy should find one
        return []
        
    return [[u, v] for u, v in best_solution]
