import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    best_n = 39
    best_edges = []
    
    def creates_c4(adj, u, v):
        au = adj[u]
        av = adj[v]
        
        if au == 0 or av == 0:
            return False
            
        temp = au
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            if adj[w] & av:
                return True
            temp ^= lsb
            
        return False

    def try_construct(n, max_attempts=50, time_limit=None):
        min_deg_req = n - 39
        if min_deg_req <= 0:
            return True, []
        
        for attempt in range(max_attempts):
            if time_limit and time.time() > time_limit:
                return False, None
            
            adj = [0] * n
            degrees = [0] * n
            
            # 1. Greedy Phase
            all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time_limit and time.time() > time_limit:
                    return False, None
                
                if not creates_c4(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    degrees[u] += 1
                    degrees[v] += 1
            
            # 2. Local Search Phase (Polish with Kicks)
            ls_deadline = time.time() + 1.0 # 1 second per attempt
            max_ls_steps = 10000
            
            for step in range(max_ls_steps):
                if time.time() > ls_deadline or (time_limit and time.time() > time_limit):
                    break
                
                # Find vertex with minimum degree
                min_deg = min(degrees)
                if min_deg >= min_deg_req:
                    break
                    
                # Pick the first vertex with min degree (or random among min)
                min_vertices = [i for i in range(n) if degrees[i] == min_deg]
                u = random.choice(min_vertices)
                
                # Try to add an edge to u
                non_neigh = [v for v in range(n) if v != u and not (adj[u] & (1 << v))]
                random.shuffle(non_neigh)
                
                added = False
                for v in non_neigh:
                    if not creates_c4(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        degrees[u] += 1
                        degrees[v] += 1
                        added = True
                        break
                        
                if not added:
                    # Kick: Remove a random edge incident to a neighbor of u
                    # This might break a C4 constraint that was blocking an edge addition
                    neigh = []
                    temp = adj[u]
                    while temp:
                        lsb = temp & -temp
                        w = lsb.bit_length() - 1
                        neigh.append(w)
                        temp ^= lsb
                        
                    if neigh:
                        # Pick a random neighbor w of u
                        w = random.choice(neigh)
                        # Remove edge (u, w)
                        adj[u] &= ~(1 << w)
                        adj[w] &= ~(1 << u)
                        degrees[u] -= 1
                        degrees[w] -= 1

            # Final Check
            min_deg = min(degrees)
            if min_deg >= min_deg_req:
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if adj[i] & (1 << j):
                            edges.append([i, j])
                return True, edges
                
        return False, None

    # Try from n=47 down to n=40
    for target_n in [47, 46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 10:
            break
            
        success, edges = try_construct(target_n, max_attempts=50, time_limit=deadline - 2)
        if success:
            best_n = target_n
            best_edges = edges
            break
            
    if best_n > 39:
        return {'n': best_n, 'edges': best_edges}
    else:
        return {'n': 39, 'edges': []}
