import time
from math import gcd

def construct():
    start_time = time.time()
    time_limit = 55.0  # Leave 5s margin for 60s limit

    N = 256
    # Build conflict graph
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            u = a // g
            v = b // g
            if (u + v) % g == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Complement graph for Max Clique
    comp_adj = [0] * N
    full_mask = (1 << N) - 1
    for i in range(N):
        # All except i
        all_except_i = full_mask ^ (1 << i)
        # Remove neighbors in original graph
        comp_adj[i] = all_except_i & ~adj[i]
        
    best_size = 0
    best_set = []
    
    # Greedy coloring for upper bound on Clique size in comp_adj
    def color_bound(P):
        colors = 0
        rem = P
        while rem:
            colors += 1
            # Pick lowest index vertex
            v = (rem & -rem).bit_length() - 1
            # Remove v and its neighbors in comp_adj
            rem &= ~(1 << v)
            rem &= ~comp_adj[v]
        return colors

    def search(P, current_clique):
        nonlocal best_size, best_set, start_time, time_limit
        
        if time.time() - start_time > time_limit:
            return
        
        if P == 0:
            if len(current_clique) > best_size:
                best_size = len(current_clique)
                best_set = list(current_clique)
            return
            
        ub = color_bound(P)
        if len(current_clique) + ub <= best_size:
            return
            
        # Heuristic: Pick vertex with max degree in P
        # To speed up, we can just pick the first one if time is tight, 
        # but max degree helps prune.
        # Let's try max degree.
        max_deg = -1
        v = -1
        temp_p = P
        while temp_p:
            u = (temp_p & -temp_p).bit_length() - 1
            temp_p &= temp_p - 1
            # Count bits in comp_adj[u] & P
            deg = (comp_adj[u] & P).bit_count()
            if deg > max_deg:
                max_deg = deg
                v = u
                
        if v == -1:
            return
            
        # Branch 1: Include v
        # New candidates are neighbors of v in comp_adj
        new_P = P & comp_adj[v]
        search(new_P, current_clique + [v])
        
        # Branch 2: Exclude v
        new_P2 = P & ~(1 << v)
        search(new_P2, current_clique)

    search(full_mask, [])
    
    result = [x + 1 for x in best_set]
    if not result:
        result = [1]
        
    return result
