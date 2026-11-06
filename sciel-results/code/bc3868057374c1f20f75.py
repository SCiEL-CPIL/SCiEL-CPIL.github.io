import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    best_n = 39
    best_edges = []
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
        
        min_deg_req = n - 39
        
        def count_c4s(adj):
            count = 0
            for u in range(n):
                for v in range(u + 1, n):
                    common = (adj[u] & adj[v]).bit_count()
                    if common >= 2:
                        count += common * (common - 1) // 2
            return count // 2
        
        def get_cost(adj):
            c4s = count_c4s(adj)
            min_deg = min(adj[u].bit_count() for u in range(n))
            deg_penalty = max(0, min_deg_req - min_deg) * 100
            return c4s + deg_penalty
        
        found_valid = False
        best_adj = None
        best_cost = float('inf')
        
        for trial in range(30):
            if time.time() > deadline:
                break
            
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.7:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            current_cost = get_cost(adj)
            best_adj = adj[:]
            best_cost = current_cost
            
            T = 50.0
            T_min = 0.01
            cooling = 0.997
            max_steps = 50000
            
            steps = 0
            while T > T_min and steps < max_steps:
                if time.time() > deadline:
                    break
                
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                
                new_adj = adj[:]
                if (adj[u] >> v) & 1:
                    new_adj[u] &= ~(1 << v)
                    new_adj[v] &= ~(1 << u)
                else:
                    new_adj[u] |= (1 << v)
                    new_adj[v] |= (1 << u)
                
                new_cost = get_cost(new_adj)
                delta = new_cost - current_cost
                
                if delta < 0 or random.random() < math.exp(-delta / T):
                    adj = new_adj
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_adj = adj[:]
                    
                    if current_cost == 0:
                        found_valid = True
                        break
                
                T *= cooling
                steps += 1
            
            if found_valid:
                break
        
        if found_valid and best_cost == 0:
            edges_G = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (best_adj[u] >> v) & 1:
                        edges_G.append([u, v])
            
            min_deg = min(best_adj[u].bit_count() for u in range(n))
            if min_deg < min_deg_req:
                continue
            
            has_c4 = False
            for u in range(n):
                for v in range(u + 1, n):
                    common = (best_adj[u] & best_adj[v]).bit_count()
                    if common >= 2:
                        has_c4 = True
                        break
                if has_c4:
                    break
            
            if not has_c4:
                return {'n': n, 'edges': edges_G}
    
    return {'n': 39, 'edges': []}
