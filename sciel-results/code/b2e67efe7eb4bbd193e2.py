import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 100.0  # Safety margin
    
    n = 16
    target_edges = 65
    
    # Initialize with K_{8,8} + (0,1)
    # 0-7 in A, 8-15 in B
    adj = [0] * n
    for u in range(8):
        for v in range(8, 16):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
    adj[0] |= (1 << 1)
    adj[1] |= (1 << 0)
    
    def get_degrees(adj):
        return [bin(x).count('1') for x in adj]
    
    def calc_score(adj):
        degs = get_degrees(adj)
        max_sum = 0
        # Iterate all triples u < v < w
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    for w in range(v + 1, n):
                        # Check if (u,v,w) is a triangle
                        if (adj[v] & (1 << w)) and (adj[w] & (1 << u)):
                            s = degs[u] + degs[v] + degs[w]
                            if s > max_sum:
                                max_sum = s
        return max_sum

    current_adj = adj[:]
    current_score = calc_score(current_adj)
    best_adj = current_adj[:]
    best_score = current_score
    
    # List of all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Helper to get current edge set as a set of tuples for fast lookup
    def get_edge_set(adj):
        es = set()
        for u in range(n):
            mask = adj[u]
            while mask:
                lsb = mask & -mask
                v = lsb.bit_length() - 1
                if v > u:
                    es.add((u, v))
                mask -= lsb
        return es

    current_edges = get_edge_set(current_adj)
    
    # SA Parameters
    T = 1.0
    T_min = 0.01
    cooling = 0.99
    
    while time.time() - start_time < time_limit:
        # Generate move: Swap an edge
        # Pick random existing edge
        if not current_edges:
            break
        e1 = random.choice(list(current_edges))
        
        # Pick random non-existing edge
        # To avoid infinite loop if graph is complete (impossible here), 
        # we can just pick random pairs and check.
        while True:
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2:
                continue
            if u2 > v2:
                u2, v2 = v2, u2
            e2 = (u2, v2)
            if e2 not in current_edges:
                break
        
        # Apply swap
        # Remove e1
        u1, v1 = e1
        current_adj[u1] &= ~(1 << v1)
        current_adj[v1] &= ~(1 << u1)
        current_edges.remove(e1)
        
        # Add e2
        current_adj[u2] |= (1 << v2)
        current_adj[v2] |= (1 << u2)
        current_edges.add(e2)
        
        new_score = calc_score(current_adj)
        
        # Acceptance
        if new_score < current_score:
            accept = True
        elif new_score == current_score:
            accept = random.random() < 0.5
        else:
            accept = random.random() < np.exp(-(new_score - current_score) / T)
        
        if accept:
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_adj = current_adj[:]
        else:
            # Revert
            current_adj[u1] |= (1 << v1)
            current_adj[v1] |= (1 << u1)
            current_edges.add(e1)
            
            current_adj[u2] &= ~(1 << v2)
            current_adj[v2] &= ~(1 << u2)
            current_edges.remove(e2)
            
        T *= cooling
        if T < T_min:
            T = T_min

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        mask = best_adj[u]
        while mask:
            lsb = mask & -mask
            v = lsb.bit_length() - 1
            if v > u:
                edges.append([u, v])
            mask -= lsb
            
    return {'n': 16, 'edges': edges}
