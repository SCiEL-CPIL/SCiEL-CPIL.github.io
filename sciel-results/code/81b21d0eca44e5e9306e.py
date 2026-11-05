import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    for n in range(47, 38, -1):
        if time.time() > deadline:
            break
            
        target_min_deg = n - 39
        if target_min_deg < 0:
            target_min_deg = 0
            
        # SA for this n
        # We try a few restarts
        for r in range(20):
            if time.time() > deadline:
                break
                
            edges = set()
            adj = [set() for _ in range(n)]
            
            # Init random
            p = 0.2
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        edges.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
            
            # Init metrics
            c4_count = 0
            for u, v in edges:
                k = len(adj[u].intersection(adj[v]))
                if k >= 2:
                    c4_count += k * (k - 1) // 2
            c4_count //= 4
            
            deg_deficit = 0
            for u in range(n):
                d = len(adj[u])
                if d < target_min_deg:
                    deg_deficit += (target_min_deg - d) ** 2
            
            T = 50.0
            cooling = 0.9995
            min_T = 0.1
            steps = 0
            max_steps = 20000
            
            while T > min_T and steps < max_steps:
                if time.time() > deadline:
                    break
                    
                move_type = random.choice(['add', 'remove', 'swap'])
                
                if move_type == 'add':
                    u = random.randint(0, n - 1)
                    v = random.randint(0, n - 1)
                    if u == v or (u, v) in edges or (v, u) in edges:
                        continue
                    
                    k = len(adj[u].intersection(adj[v]))
                    delta_c4 = k * (k - 1) // 2
                    
                    # Degree change
                    delta_def = 0
                    for x in [u, v]:
                        old_d = len(adj[x])
                        new_d = old_d + 1
                        old_def = (target_min_deg - old_d) ** 2 if old_d < target_min_deg else 0
                        new_def = (target_min_deg - new_d) ** 2 if new_d < target_min_deg else 0
                        delta_def += new_def - old_def
                    
                    delta_e = delta_c4 * 1000 + delta_def
                    if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                        edges.add((u, v))
                        adj[u].add(v)
                        adj[v].add(u)
                        c4_count += delta_c4
                        deg_deficit += delta_def
                        
                elif move_type == 'remove':
                    if not edges:
                        continue
                    u, v = random.choice(list(edges))
                    k = len(adj[u].intersection(adj[v]))
                    delta_c4 = - (k * (k - 1) // 2)
                    
                    delta_def = 0
                    for x in [u, v]:
                        old_d = len(adj[x])
                        new_d = old_d - 1
                        old_def = (target_min_deg - old_d) ** 2 if old_d < target_min_deg else 0
                        new_def = (target_min_deg - new_d) ** 2 if new_d < target_min_deg else 0
                        delta_def += new_def - old_def
                    
                    delta_e = delta_c4 * 1000 + delta_def
                    if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                        edges.remove((u, v))
                        adj[u].remove(v)
                        adj[v].remove(u)
                        c4_count += delta_c4
                        deg_deficit += delta_def
                        
                else: # swap
                    u = random.randint(0, n - 1)
                    if not adj[u]:
                        continue
                    v = random.choice(list(adj[u]))
                    w = random.randint(0, n - 1)
                    if w == u or w == v or w in adj[u]:
                        continue
                    
                    # Remove (u,v), Add (u,w)
                    # 1. Remove (u,v)
                    k_uv = len(adj[u].intersection(adj[v]))
                    delta_c4_1 = - (k_uv * (k_uv - 1) // 2)
                    
                    # 2. Add (u,w)
                    # Note: adj[u] and adj[w] are still as before (v is in adj[u])
                    # But we are removing v from u.
                    # Common neighbors of u and w:
                    # N(u) \ {v} intersect N(w)
                    # Since v is not in N(w) (w not in adj[u] implies u not in adj[w], so v!=w, and w not adj u),
                    # and v is in N(u).
                    # Does v share edge with w?
                    # If (v,w) in edges, then v is common neighbor of u and w?
                    # u~v, u~w. v~w?
                    # If v~w, then v is in N(u) and N(w).
                    # So k_uw = len( (adj[u] - {v}) .intersection(adj[w]) )
                    k_uw = len((adj[u] - {v}).intersection(adj[w]))
                    delta_c4_2 = k_uw * (k_uw - 1) // 2
                    
                    delta_c4 = delta_c4_1 + delta_c4_2
                    
                    # Degree change
                    # u: deg unchanged
                    # v: deg -1
                    # w: deg +1
                    delta_def = 0
                    # v
                    old_d_v = len(adj[v])
                    new_d_v = old_d_v - 1
                    old_def_v = (target_min_deg - old_d_v) ** 2 if old_d_v < target_min_deg else 0
                    new_def_v = (target_min_deg - new_d_v) ** 2 if new_d_v < target_min_deg else 0
                    delta_def += new_def_v - old_def_v
                    
                    # w
                    old_d_w = len(adj[w])
                    new_d_w = old_d_w + 1
                    old_def_w = (target_min_deg - old_d_w) ** 2 if old_d_w < target_min_deg else 0
                    new_def_w = (target_min_deg - new_d_w) ** 2 if new_d_w < target_min_deg else 0
                    delta_def += new_def_w - old_def_w
                    
                    delta_e = delta_c4 * 1000 + delta_def
                    if delta_e < 0 or random.random() < pow(2.71828, -delta_e / T):
                        edges.remove((u, v))
                        adj[u].remove(v)
                        adj[v].remove(u)
                        
                        edges.add((u, w))
                        adj[u].add(w)
                        adj[w].add(u)
                        
                        c4_count += delta_c4
                        deg_deficit += delta_def
                        
                T *= cooling
                steps += 1
                
                if c4_count == 0 and deg_deficit == 0:
                    edge_list = [[u, v] for u, v in edges]
                    return {'n': n, 'edges': edge_list}
            
    return {'n': 39, 'edges': []}
