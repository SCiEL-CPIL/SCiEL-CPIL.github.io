import random
import time
import sys

def get_max_triangle_free(adj):
    n = len(adj)
    # Precompute list of vertices for each subset? No, just use bit masks.
    # We want the largest k such that there is a subset of size k with no triangles.
    
    def is_triangle_free(mask):
        # Check if subgraph induced by mask is triangle-free
        # For each u in mask, check if neighbors of u in mask form an independent set
        m = mask
        while m:
            lsb = m & -m
            u = lsb.bit_length() - 1
            m -= lsb
            # Neighbors of u in mask
            nu = adj[u] & mask
            # Check if nu has any edges
            # For each v in nu, check if adj[v] & nu is non-zero
            nn = nu
            while nn:
                lsb2 = nn & -nn
                v = lsb2.bit_length() - 1
                nn -= lsb2
                if (adj[v] & nu) != 0:
                    return False
        return True

    # Iterate k from n down to 1
    # To speed up, we can just check all subsets and track max size?
    # No, checking all 2^18 subsets is 262k.
    # For each, is_triangle_free is fast.
    # Let's just iterate all masks and find the max popcount that is triangle-free.
    max_size = 0
    for mask in range(1, 1 << n):
        if is_triangle_free(mask):
            size = bin(mask).count('1')
            if size > max_size:
                max_size = size
                # If we found n, we can stop? No, we want max.
                # If max_size == n, break? No, n is max possible.
                # But we iterate up. So if we find n, it's the best.
                # However, we iterate 1 to 2^n-1.
                # We can iterate by size?
                # Iterating by size allows early termination if we find one.
                # But generating subsets of size k is more complex.
                # Given 2^18 is small, full scan is fine.
    return max_size

def is_k4_free(adj):
    n = len(adj)
    for u in range(n):
        for v in range(u+1, n):
            if (adj[u] >> v) & 1:
                common = adj[u] & adj[v]
                # Check if common is independent
                c = common
                while c:
                    lsb = c & -c
                    x = lsb.bit_length() - 1
                    c -= lsb
                    if (adj[x] & common) != 0:
                        return False
    return True

def construct():
    n = 18
    # Start with K6,6,6
    # Parts: 0-5, 6-11, 12-17
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            part_i = i // 6
            part_j = j // 6
            if part_i != part_j:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    best_score = get_max_triangle_free(adj)
    best_adj = adj[:]
    
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Local Search
    T = 1.0
    for step in range(5000):
        if time.time() > deadline:
            break
            
        # Pick a random move
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
            
        # Toggle edge
        new_adj = adj[:]
        if (new_adj[u] >> v) & 1:
            new_adj[u] &= ~(1 << v)
            new_adj[v] &= ~(1 << u)
        else:
            new_adj[u] |= (1 << v)
            new_adj[v] |= (1 << u)
            
        if not is_k4_free(new_adj):
            continue
            
        new_score = get_max_triangle_free(new_adj)
        
        if new_score < best_score:
            best_score = new_score
            best_adj = new_adj
            adj = new_adj
            T = 1.0
        else:
            # Accept with probability
            delta = new_score - best_score
            if delta > 0:
                prob = 1.0 / (1.0 + step/100.0) # Simple cooling
                if random.random() < prob:
                    adj = new_adj
            else:
                adj = new_adj
                
        if best_score <= 9:
            break
            
    # Convert best_adj to edge list
    edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
    return edges
