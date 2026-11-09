import random
import time
import math

def construct():
    n = 20
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    start_time = time.time()
    time_limit = 115.0
    
    best_edges_global = []
    best_count_global = 0
    
    def count_violations(C):
        viol = 0
        for u in range(n):
            for v in range(u+1, n):
                if C[u][v] > 2:
                    viol += 1
        return viol

    def init_greedy_valid():
        adj = [[0]*n for _ in range(n)]
        C = [[0]*n for _ in range(n)]
        edges = set()
        shuffled_edges = all_edges[:]
        random.shuffle(shuffled_edges)
        
        for u, v in shuffled_edges:
            if C[u][v] > 2:
                continue
            valid = True
            for w in range(n):
                if w == u or w == v:
                    continue
                # If we add (u,v), C[u][w] becomes C[u][w] + adj[v][w]
                # We need C[u][w] + adj[v][w] <= 2
                if C[u][w] + adj[v][w] > 2:
                    valid = False
                    break
                if C[v][w] + adj[u][w] > 2:
                    valid = False
                    break
            if valid:
                edges.add((u, v))
                adj[u][v] = 1
                adj[v][u] = 1
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[u][w]:
                        C[v][w] += 1
                        C[w][v] += 1
                    if adj[v][w]:
                        C[u][w] += 1
                        C[w][u] += 1
        return adj, C, edges

    def run_sa(adj_init, C_init, edges_init, t_start, t_end, steps, penalty):
        adj = [row[:] for row in adj_init]
        C = [row[:] for row in C_init]
        edges = set(edges_init)
        
        current_edges_count = len(edges)
        current_violations = count_violations(C)
        current_energy = -current_edges_count + penalty * current_violations
        
        best_adj_sa = [row[:] for row in adj]
        best_C_sa = [row[:] for row in C]
        best_edges_sa = set(edges)
        best_energy_sa = current_energy
        
        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            progress = step / steps
            T = t_start - (t_start - t_end) * progress
            if T < 1e-6: T = 1e-6
            
            move_type = random.random()
            
            if move_type < 0.5:
                # Add Edge
                u, v = random.choice(all_edges)
                if (u, v) in edges:
                    continue
                
                delta_viol = 0
                # Check impact on (u,w) and (v,w)
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[v][w] == 1:
                        if C[u][w] == 2:
                            delta_viol += 1
                    if adj[u][w] == 1:
                        if C[v][w] == 2:
                            delta_viol += 1
                            
                delta_edges = 1
                delta_energy = -delta_edges + penalty * delta_viol
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    edges.add((u, v))
                    adj[u][v] = 1
                    adj[v][u] = 1
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[u][w]:
                            C[v][w] += 1
                            C[w][v] += 1
                        if adj[v][w]:
                            C[u][w] += 1
                            C[w][u] += 1
                    current_edges_count += 1
                    current_violations += delta_viol
                    current_energy += delta_energy
                    
                    if current_energy < best_energy_sa:
                        best_energy_sa = current_energy
                        best_adj_sa = [row[:] for row in adj]
                        best_C_sa = [row[:] for row in C]
                        best_edges_sa = set(edges)
                        
            else:
                # Remove Edge
                if not edges:
                    continue
                u, v = random.choice(list(edges))
                
                delta_viol = 0
                for w in range(n):
                    if w == u or w == v:
                        continue
                    if adj[v][w] == 1:
                        if C[u][w] > 2:
                            # New C[u][w] = C[u][w] - 1
                            if C[u][w] - 1 <= 2:
                                delta_viol -= 1
                    if adj[u][w] == 1:
                        if C[v][w] > 2:
                            if C[v][w] - 1 <= 2:
                                delta_viol -= 1

                delta_edges = -1
                delta_energy = -delta_edges + penalty * delta_viol
                
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / T):
                    edges.remove((u, v))
                    adj[u][v] = 0
                    adj[v][u] = 0
                    for w in range(n):
                        if w == u or w == v:
                            continue
                        if adj[u][w]:
                            C[v][w] -= 1
                            C[w][v] -= 1
                        if adj[v][w]:
                            C[u][w] -= 1
                            C[w][u] -= 1
                    current_edges_count -= 1
                    current_violations += delta_viol
                    current_energy += delta_energy
                    
                    if current_energy < best_energy_sa:
                        best_energy_sa = current_energy
                        best_adj_sa = [row[:] for row in adj]
                        best_C_sa = [row[:] for row in C]
                        best_edges_sa = set(edges)

        return best_edges_sa, best_C_sa

    # Run multiple SA instances
    # We can also vary the penalty
    penalties = [5.0, 10.0, 20.0]
    
    for p in penalties:
        if time.time() - start_time > time_limit:
            break
            
        # Initialize with greedy
        adj, C, edges = init_greedy_valid()
        
        # Update global best with greedy
        if len(edges) > best_count_global:
            best_count_global = len(edges)
            best_edges_global = [list(e) for e in edges]
            
        # Run SA
        best_edges_sa, best_C_sa = run_sa(adj, C, edges, 1.0, 0.01, 3000, p)
        
        # Check validity of SA result
        if count_violations(best_C_sa) == 0:
            if len(best_edges_sa) > best_count_global:
                best_count_global = len(best_edges_sa)
                best_edges_global = [list(e) for e in best_edges_sa]
        else:
            # Try to repair by removing edges from best_edges_sa that cause violations
            # This is a fallback
            pass

    return best_edges_global
