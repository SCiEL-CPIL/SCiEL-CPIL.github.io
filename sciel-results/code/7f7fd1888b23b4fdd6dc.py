import time
import random

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        edge_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    edge_count += 1
        return adj, edge_count

    def count_bits(x):
        return bin(x).count("1")

    def greedy_clique_partition(adj, n, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random.Random()
            
        # Copy adj to remaining_adj
        rem_adj = adj[:]
        cliques = []
        
        # Precompute degrees? No, update dynamically or just use count_bits.
        
        while True:
            # Find vertex with max degree
            max_deg = 0
            start_v = -1
            for u in range(n):
                deg = count_bits(rem_adj[u])
                if deg > max_deg:
                    max_deg = deg
                    start_v = u
            if max_deg == 0:
                break
            
            # Find a large clique starting from start_v
            clique = [start_v]
            candidates = rem_adj[start_v]
            
            while candidates:
                # Pick candidate with max degree in remaining graph
                best_cand = -1
                best_deg = -1
                c = candidates
                while c:
                    # Get lowest set bit
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    deg = count_bits(rem_adj[v])
                    if deg > best_deg:
                        best_deg = deg
                        best_cand = v
                    c ^= lsb
                
                if best_cand == -1:
                    break
                
                clique.append(best_cand)
                # Update candidates: intersection with neighbors of best_cand
                candidates &= rem_adj[best_cand]
            
            # Remove edges in clique
            for u in clique:
                for v in clique:
                    if u < v:
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
        
        return cliques

    start_time = time.time()
    deadline = start_time + 110
    
    adj_1, _ = instance(32, 0.6, 601)
    adj_2, _ = instance(64, 0.55, 602)
    
    best_cliques_1 = None
    best_cliques_2 = None
    best_len_1 = float('inf')
    best_len_2 = float('inf')
    
    seed_counter = 0
    while time.time() < deadline:
        # Run greedy for instance 1
        cliques_1 = greedy_clique_partition(adj_1, 32, seed=seed_counter)
        if len(cliques_1) < best_len_1:
            best_len_1 = len(cliques_1)
            best_cliques_1 = cliques_1
        
        # Run greedy for instance 2
        cliques_2 = greedy_clique_partition(adj_2, 64, seed=seed_counter)
        if len(cliques_2) < best_len_2:
            best_len_2 = len(cliques_2)
            best_cliques_2 = cliques_2
        
        seed_counter += 1
        if seed_counter % 50 == 0:
             if time.time() > deadline:
                 break

    if best_cliques_1 is None:
        # Fallback: individual edges
        # Reconstruct edges from adj_1
        cliques = []
        for u in range(32):
            for v in range(u+1, 32):
                if adj_1[u] & (1 << v):
                    cliques.append([u, v])
        best_cliques_1 = cliques
        
    if best_cliques_2 is None:
        cliques = []
        for u in range(64):
            for v in range(u+1, 64):
                if adj_2[u] & (1 << v):
                    cliques.append([u, v])
        best_cliques_2 = cliques

    return [best_cliques_1, best_cliques_2]
