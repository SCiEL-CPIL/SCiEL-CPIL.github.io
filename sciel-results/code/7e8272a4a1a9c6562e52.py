import random
import time
import math

def construct():
    n = 16
    target_edges = 65
    
    def get_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v]:
                    edges.append((u, v))
        return edges

    def get_non_edges(adj):
        non_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    non_edges.append((u, v))
        return non_edges

    def get_degrees(adj):
        degs = [0]*n
        for u in range(n):
            for v in range(n):
                if adj[u][v]:
                    degs[u] += 1
        return degs
    
    def calc_score(adj, degs):
        max_sum = 0
        for u in range(n):
            for v in range(u+1, n):
                if not adj[u][v]:
                    continue
                for w in range(v+1, n):
                    if adj[u][w] and adj[v][w]:
                        s = degs[u] + degs[v] + degs[w]
                        if s > max_sum:
                            max_sum = s
        return max_sum

    def build_k88_plus_one():
        adj = [[False]*n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = adj[v][u] = True
        adj[0][1] = adj[1][0] = True
        return adj

    def local_search(adj, max_time=5.0):
        """Exhaustive local search for single edge swaps"""
        start = time.time()
        degs = get_degrees(adj)
        score = calc_score(adj, degs)
        
        while time.time() - start < max_time:
            edges = get_edges(adj)
            non_edges = get_non_edges(adj)
            
            best_swap = None
            best_score = score
            
            for rem_u, rem_v in edges:
                for add_u, add_v in non_edges:
                    if (rem_u, rem_v) == (add_u, add_v):
                        continue
                    
                    adj[rem_u][rem_v] = adj[rem_v][rem_u] = False
                    adj[add_u][add_v] = adj[add_v][add_u] = True
                    
                    new_degs = degs[:]
                    new_degs[rem_u] -= 1
                    new_degs[rem_v] -= 1
                    new_degs[add_u] += 1
                    new_degs[add_v] += 1
                    
                    new_score = calc_score(adj, new_degs)
                    
                    adj[rem_u][rem_v] = adj[rem_v][rem_u] = True
                    adj[add_u][add_v] = adj[add_v][add_u] = False
                    
                    if new_score < best_score:
                        best_score = new_score
                        best_swap = (rem_u, rem_v, add_u, add_v)
            
            if best_swap is None:
                break
            
            rem_u, rem_v, add_u, add_v = best_swap
            adj[rem_u][rem_v] = adj[rem_v][rem_u] = False
            adj[add_u][add_v] = adj[add_v][add_u] = True
            
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            degs[add_u] += 1
            degs[add_v] += 1
            
            score = best_score
            
            if score < 24:
                break
                
        return adj, score

    def simulated_annealing(adj, start_time, max_time=10.0):
        degs = get_degrees(adj)
        current_score = calc_score(adj, degs)
        best_score = current_score
        best_adj = [row[:] for row in adj]
        
        T = 2.0
        cooling = 0.99
        min_T = 0.05
        
        edges = get_edges(adj)
        non_edges = get_non_edges(adj)
        
        while time.time() - start_time < max_time and T > min_T:
            if not edges or not non_edges:
                break
                
            rem_u, rem_v = random.choice(edges)
            add_u, add_v = random.choice(non_edges)
            
            if (rem_u, rem_v) == (add_u, add_v):
                continue
            
            adj[rem_u][rem_v] = adj[rem_v][rem_u] = False
            adj[add_u][add_v] = adj[add_v][add_u] = True
            
            degs[rem_u] -= 1
            degs[rem_v] -= 1
            degs[add_u] += 1
            degs[add_v] += 1
            
            new_score = calc_score(adj, degs)
            
            if new_score <= current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_adj = [row[:] for row in adj]
                # Update edge lists
                edges.remove((rem_u, rem_v))
                non_edges.append((rem_u, rem_v))
                non_edges.remove((add_u, add_v))
                edges.append((add_u, add_v))
            else:
                delta = new_score - current_score
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_score = new_score
                    # Update edge lists
                    edges.remove((rem_u, rem_v))
                    non_edges.append((rem_u, rem_v))
                    non_edges.remove((add_u, add_v))
                    edges.append((add_u, add_v))
                else:
                    # Revert
                    adj[rem_u][rem_v] = adj[rem_v][rem_u] = True
                    adj[add_u][add_v] = adj[add_v][add_u] = False
                    degs[rem_u] += 1
                    degs[rem_v] += 1
                    degs[add_u] -= 1
                    degs[add_v] -= 1
            
            T *= cooling
            
        return best_adj, best_score

    def construct_hybrid():
        total_start = time.time()
        max_total_time = 115.0
        
        best_adj = None
        best_score = float('inf')
        
        # 1. Start with K_{8,8}+1
        init_adj = build_k88_plus_one()
        adj_copy, sa_score = simulated_annealing(init_adj, total_start, max_time=10.0)
        
        # 2. Polish with local search
        if time.time() - total_start < max_total_time:
            adj_copy, ls_score = local_search(adj_copy, max_time=10.0)
            if ls_score < best_score:
                best_score = ls_score
                best_adj = adj_copy
        
        # 3. Try random restarts with SA + LS
        while time.time() - total_start < max_total_time - 20:
            # Random graph init
            adj_rand = [[False]*n for _ in range(n)]
            all_edges = [(u, v) for u in range(n) for v in range(u+1, n)]
            random.shuffle(all_edges)
            for u, v in all_edges[:target_edges]:
                adj_rand[u][v] = adj_rand[v][u] = True
            
            time_left = max_total_time - (time.time() - total_start)
            if time_left < 10:
                break
                
            sa_time = min(5.0, time_left * 0.4)
            ls_time = min(5.0, time_left * 0.4)
            
            adj_sa, _ = simulated_annealing(adj_rand, total_start, max_time=sa_time)
            
            if time.time() - total_start < max_total_time - ls_time:
                adj_ls, score = local_search(adj_sa, max_time=ls_time)
                if score < best_score:
                    best_score = score
                    best_adj = adj_ls
        
        if best_adj is None:
            best_adj = build_k88_plus_one()
            best_score = 24

        final_edges = []
        for u in range(n):
            for v in range(u+1, n):
                if best_adj[u][v]:
                    final_edges.append([u, v])
        
        return {'n': n, 'edges': final_edges}

    return construct_hybrid()
