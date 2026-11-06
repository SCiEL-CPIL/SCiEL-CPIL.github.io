import random
import time
import math
import copy

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

    def random_graph():
        adj = [[False]*n for _ in range(n)]
        all_possible = [(u, v) for u in range(n) for v in range(u+1, n)]
        chosen = random.sample(all_possible, target_edges)
        for u, v in chosen:
            adj[u][v] = True
            adj[v][u] = True
        return adj

    def simulated_annealing(adj_init, max_time=10.0):
        adj = copy.deepcopy(adj_init)
        degs = get_degrees(adj)
        score = calc_score(adj, degs)
        
        # Precompute potential non-edges and edges for faster swap selection?
        # For n=16, full scan is fast enough.
        
        start_time = time.time()
        T = 10.0
        cooling = 0.9995
        
        while time.time() - start_time < max_time:
            # Generate a move: remove edge (u1, v1), add edge (u2, v2)
            # To ensure we don't pick the same edge, we pick a random edge and a random non-edge
            # If they are the same, retry (or just pick another non-edge)
            
            edges = get_edges(adj)
            non_edges = []
            for u in range(n):
                for v in range(u+1, n):
                    if not adj[u][v]:
                        non_edges.append((u, v))
            
            if not edges or not non_edges:
                break
                
            rem_u, rem_v = random.choice(edges)
            add_u, add_v = random.choice(non_edges)
            
            # If we picked the same edge (shouldn't happen if disjoint sets, but just in case)
            if (rem_u, rem_v) == (add_u, add_v):
                continue
                
            # Apply move
            adj[rem_u][rem_v] = adj[rem_v][rem_u] = False
            adj[add_u][add_v] = adj[add_v][add_u] = True
            
            new_degs = degs[:]
            new_degs[rem_u] -= 1
            new_degs[rem_v] -= 1
            new_degs[add_u] += 1
            new_degs[add_v] += 1
            
            new_score = calc_score(adj, new_degs)
            
            delta = new_score - score
            
            if delta < 0:
                # Improvement
                score = new_score
                degs = new_degs
            else:
                # Worse, accept with probability exp(-delta/T)
                if random.random() < math.exp(-delta / T):
                    score = new_score
                    degs = new_degs
                else:
                    # Revert
                    adj[rem_u][rem_v] = adj[rem_v][rem_u] = True
                    adj[add_u][add_v] = adj[add_v][add_u] = False
            
            T *= cooling
            
            if score == 21: # Theoretical lower bound is around 23, 21 is very good
                break
        return adj, score

    # Run multiple restarts
    total_start = time.time()
    best_score = float('inf')
    best_adj = None
    
    num_restarts = 10
    time_per_restart = 8.0
    
    for i in range(num_restarts):
        if time.time() - total_start > 100:
            break
            
        init_adj = random_graph()
        # Optional: Quick local search to start SA from a better point? 
        # Or just start random. Random is fine for SA.
        
        adj, score = simulated_annealing(init_adj, max_time=time_per_restart)
        
        if score < best_score:
            best_score = score
            best_adj = adj
        
        if best_score <= 23:
            break

    # If SA failed to find a good graph, fallback to a simple construction
    # K_{8,8} + 1 edge has score 24.
    if best_score > 24:
        adj = [[False]*n for _ in range(n)]
        for u in range(8):
            for v in range(8, 16):
                adj[u][v] = adj[v][u] = True
        adj[0][1] = adj[1][0] = True
        best_adj = adj
        best_score = 24

    # Convert to edge list
    final_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v]:
                final_edges.append([u, v])
    
    return {'n': n, 'edges': final_edges}
