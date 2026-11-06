import time
import numpy as np
from random import Random

def construct():
    def generate_graph(n, p, seed):
        rng = Random(seed)
        adj = np.zeros((n, n), dtype=bool)
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u, v] = True
                    adj[v, u] = True
        return adj

    def solve_graph(adj):
        n = adj.shape[0]
        remaining_adj = adj.copy()
        cliques = []
        
        deadline = time.time() + 110.0
        
        while True:
            if time.time() > deadline:
                break
            
            if not np.any(remaining_adj):
                break
                
            degrees = remaining_adj.sum(axis=1)
            v = int(np.argmax(degrees))
            
            if degrees[v] == 0:
                break
            
            C = [v]
            candidates = np.where(remaining_adj[v])[0]
            
            while len(candidates) > 0:
                cand_degrees = remaining_adj[candidates].sum(axis=1)
                u_idx = int(np.argmax(cand_degrees))
                u = candidates[u_idx]
                
                C.append(u)
                
                mask = remaining_adj[u][candidates]
                candidates = candidates[mask]
                
            if len(C) >= 2:
                cliques.append(C)
            
            for i in range(len(C)):
                for j in range(i + 1, len(C)):
                    u, w = C[i], C[j]
                    remaining_adj[u, w] = False
                    remaining_adj[w, u] = False

        return cliques

    n1, p1, seed1 = 32, 0.6, 601
    adj1 = generate_graph(n1, p1, seed1)
    cliques1 = solve_graph(adj1)
    
    n2, p2, seed2 = 64, 0.55, 602
    adj2 = generate_graph(n2, p2, seed2)
    cliques2 = solve_graph(adj2)
    
    return [cliques1, cliques2]
