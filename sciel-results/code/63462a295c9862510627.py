import time
import random
import math

def construct():
    start_time = time.time()
    deadline = 148
    
    n = 16
    adj = [[False]*n for _ in range(n)]
    edge_set = set()
    
    def add_edge(u, v):
        u, v = min(u,v), max(u,v)
        if u != v and (u,v) not in edge_set:
            edge_set.add((u,v))
            adj[u][v] = adj[v][u] = True
    
    def remove_edge(u, v):
        u, v = min(u,v), max(u,v)
        if (u,v) in edge_set:
            edge_set.remove((u,v))
            adj[u][v] = adj[v][u] = False
    
    def compute_score():
        deg = [0]*n
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    deg[i] += 1
                    deg[j] += 1
        max_sum = 0
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            s = deg[i] + deg[j] + deg[k]
                            if s > max_sum:
                                max_sum = s
        return max_sum
    
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    
    best_score = float('inf')
    best_edge_set = None
    
    def try_local_search(init_edge_set, max_iterations=100000):
        nonlocal best_score, best_edge_set
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        for e in init_edge_set:
            add_edge(*e)
        
        current_score = compute_score()
        
        if current_score < best_score:
            best_score = current_score
            best_edge_set = set(edge_set)
        
        no_improve = 0
        for it in range(max_iterations):
            if time.time() > deadline - 3:
                break
            
            improved = False
            # Try all possible single swaps
            for e_remove in list(edge_set):
                if time.time() > deadline - 3:
                    break
                for e_add in all_edges:
                    if e_add in edge_set or e_add == e_remove:
                        continue
                    
                    remove_edge(*e_remove)
                    add_edge(*e_add)
                    
                    new_score = compute_score()
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_edge_set = set(edge_set)
                        break
                    else:
                        remove_edge(*e_add)
                        add_edge(*e_remove)
                
                if improved:
                    break
            
            if not improved:
                # Try 2-swap moves
                improved2 = False
                edges_list = list(edge_set)
                for i in range(len(edges_list)):
                    if time.time() > deadline - 3:
                        break
                    for j in range(i+1, len(edges_list)):
                        if time.time() > deadline - 3:
                            break
                        e_rem1, e_rem2 = edges_list[i], edges_list[j]
                        for e_add1 in all_edges:
                            if time.time() > deadline - 3:
                                break
                            if e_add1 in edge_set or e_add1 == e_rem1 or e_add1 == e_rem2:
                                continue
                            for e_add2 in all_edges:
                                if e_add2 in edge_set or e_add2 == e_rem1 or e_add2 == e_rem2 or e_add2 == e_add1:
                                    continue
                                
                                remove_edge(*e_rem1)
                                remove_edge(*e_rem2)
                                add_edge(*e_add1)
                                add_edge(*e_add2)
                                
                                new_score = compute_score()
                                
                                if new_score < current_score:
                                    current_score = new_score
                                    improved2 = True
                                    if current_score < best_score:
                                        best_score = current_score
                                        best_edge_set = set(edge_set)
                                    break
                                else:
                                    remove_edge(*e_add1)
                                    remove_edge(*e_add2)
                                    add_edge(*e_rem1)
                                    add_edge(*e_rem2)
                                
                                if improved2:
                                    break
                            
                            if improved2:
                                break
                        
                        if improved2:
                            break
                    
                    if improved2:
                        break
                
                if improved2:
                    no_improve = 0
                    continue
                else:
                    no_improve += 1
                    if no_improve > 50:
                        break
    
    # Generate multiple diverse initial configurations
    initial_configs = []
    
    # Config 1: K_{8,8} + one edge in part 0
    for u in range(8):
        for v in range(u+1, 8):
            base = set((a,b) for a in range(8) for b in range(8,16))
            initial_configs.append(base | {(u,v)})
    
    # Config 2: K_{8,8} + one edge in part 1
    for u in range(8):
        for v in range(u+1, 8):
            base = set((a,b) for a in range(8) for b in range(8,16))
            initial_configs.append(base | {(u+8, v+8)})
    
    # Config 3: K_{7,9} + edges to make 65
    # K_{7,9} has 63 edges, need 2 more
    for u in range(7):
        for v in range(u+1, 7):
            base = set((a,b) for a in range(7) for b in range(7,16))
            # Add 2 edges within part 0
            for w in range(v+1, 7):
                initial_configs.append(base | {(u,v), (v,w)})
    
    # Shuffle and try a subset
    random.seed(42)
    random.shuffle(initial_configs)
    
    # Try top configurations
    tried = 0
    for config in initial_configs:
        if time.time() > deadline - 15:
            break
        try_local_search(config, max_iterations=50000)
        tried += 1
        if tried >= 20:
            break
    
    if best_edge_set is None:
        edge_set.clear()
        for i in range(n):
            adj[i] = [False]*n
        for u in range(8):
            for v in range(8, 16):
                add_edge(u, v)
        add_edge(0, 1)
        best_edge_set = set(edge_set)
    
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}
