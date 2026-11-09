import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    n = 16
    
    # Construct the Clebsch graph (Folded 5-cube)
    # Vertices are binary strings of length 5, identified with their complements.
    # We can represent them by the first 4 bits, or simply 0..15.
    # Actually, easier: Vertices 0..15 (4 bits).
    # Edge (u, v) if Hamming distance of 5-bit representations is 1.
    # Representation: u is 0..15. 5-bit rep is u (0000u) and complement (1111u^15).
    # Two vertices u, v are adjacent if:
    # 1. d5(u, v) == 1  (where d5 is Hamming distance in 5 bits, considering u as 0000u)
    #    Wait, the folded cube vertices are pairs {x, x^c}.
    #    Let's map 0..15 to the 16 equivalence classes.
    #    Class i corresponds to binary string b (4 bits).
    #    The 5-bit representatives are 0b and 1b^15.
    #    Two classes i, j are adjacent if any representative of i is distance 1 from any representative of j.
    #    This is equivalent to:
    #    d4(i, j) == 1  (distance in 4 bits)
    #    OR
    #    d4(i, j) == 3  (distance in 4 bits, which corresponds to distance 2 in 5 bits? No)
    #    Let's check:
    #    Reps of i: 0i, 1i^15.
    #    Reps of j: 0j, 1j^15.
    #    Distances:
    #    d(0i, 0j) = d4(i,j)
    #    d(0i, 1j^15) = 1 + d4(i, j^15) = 1 + (4 - d4(i,j)) = 5 - d4(i,j)
    #    d(1i^15, 0j) = 1 + d4(i^15, j) = 1 + (4 - d4(i,j)) = 5 - d4(i,j)
    #    d(1i^15, 1j^15) = d4(i^15, j^15) = d4(i,j)
    #    So possible distances are d4(i,j) and 5-d4(i,j).
    #    We want distance 1.
    #    So d4(i,j) == 1 OR 5-d4(i,j) == 1 => d4(i,j) == 4.
    #    So edges are between i, j if d4(i,j) == 1 or d4(i,j) == 4.
    
    def hamming_distance(a, b):
        return bin(a ^ b).count('1')
        
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_distance(i, j)
            if d == 1 or d == 4:
                edges.append([i, j])
                
    # Validate
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
        
    def is_triangle_free(adj, n):
        for i in range(n):
            for j in adj[i]:
                if i > j:
                    continue
                if adj[i] & adj[j]:
                    return False
        return True
        
    def is_not_3_colorable(adj, n):
        colors = [-1] * n
        def backtrack(v):
            if v == n:
                return True
            for c in range(3):
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if backtrack(v + 1):
                        return True
                    colors[v] = -1
            return False
        order = sorted(range(n), key=lambda x: len(adj[x]), reverse=True)
        colors = [-1] * n
        def bt(idx):
            if idx == n:
                return True
            v = order[idx]
            for c in range(3):
                valid = True
                for nb in adj[v]:
                    if colors[nb] == c:
                        valid = False
                        break
                if valid:
                    colors[v] = c
                    if bt(idx + 1):
                        return True
                    colors[v] = -1
            return False
        return not bt(0)
        
    if not is_triangle_free(adj, n) or not is_not_3_colorable(adj, n):
        # Fallback to Mycielski C7 + greedy
        def get_mycielski_c7():
            n_cycle = 7
            e = []
            for i in range(n_cycle):
                e.append([i, (i + 1) % n_cycle])
            for i in range(n_cycle):
                neighbors = [(i + 1) % n_cycle, (i - 1) % n_cycle]
                for nb in neighbors:
                    e.append([7 + i, nb])
            for i in range(n_cycle):
                e.append([14, 7 + i])
            return e
        return get_mycielski_c7()
        
    return edges
