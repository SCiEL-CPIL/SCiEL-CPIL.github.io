import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0

    N = 255

    candidates = []
    for i in range(1, N + 1):
        if gcd(i, N) > 1:
            candidates.append(i)

    if N in candidates:
        candidates.remove(N)

    n_nodes = len(candidates)

    adj = [0] * n_nodes
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if gcd(candidates[i], candidates[j]) > 1:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    best_size = 0
    best_clique = []

    def greedy_clique(start_node=None):
        if start_node is None:
            start_node = random.randint(0, n_nodes - 1)
            
        clique = [start_node]
        remaining = set(range(n_nodes))
        remaining.remove(start_node)
        
        # Precompute clique mask
        clique_mask = 1 << start_node
        
        while True:
            best_next = -1
            max_deg = -1
            
            for u in remaining:
                # Check if u connects to all in clique
                if (adj[u] & clique_mask) == clique_mask:
                    # Calculate degree of u in the subgraph induced by remaining
                    # This helps pick a "central" node
                    deg = bin(adj[u] & ( (1 << n_nodes) - 1 )).count('1') 
                    # Actually, we should look at degree in the *remaining* set that are also neighbors
                    # But simple degree is a good heuristic.
                    
                    if deg > max_deg:
                        max_deg = deg
                        best_next = u
            
            if best_next != -1:
                clique.append(best_next)
                remaining.remove(best_next)
                clique_mask |= (1 << best_next)
            else:
                break
                
        return clique

    def local_search(clique):
        clique_set = set(clique)
        improved = True
        while improved:
            improved = False
            clique_mask = 0
            for v in clique:
                clique_mask |= (1 << v)
                
            for u in range(n_nodes):
                if u in clique_set:
                    continue
                if (adj[u] & clique_mask) == clique_mask:
                    clique.append(u)
                    clique_set.add(u)
                    improved = True
                    break
            
        return clique

    # Run multiple random restarts
    num_restarts = 100
    for _ in range(num_restarts):
        if time.time() > deadline:
            break
        c = greedy_clique()
        c = local_search(c)
        if len(c) > best_size:
            best_size = len(c)
            best_clique = c

    # Fallback to multiples of 3
    mult3 = [i for i in range(n_nodes) if candidates[i] % 3 == 0]
    if len(mult3) > best_size:
        best_size = len(mult3)
        best_clique = mult3

    result = [candidates[i] for i in best_clique]
    result.append(N)
    
    for i in range(len(result)):
        for j in range(i+1, len(result)):
            if gcd(result[i], result[j]) == 1:
                return list(range(3, N + 1, 3))
                
    return result
