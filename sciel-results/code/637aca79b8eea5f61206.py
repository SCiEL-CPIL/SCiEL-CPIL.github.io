"""Program for hexagon-free graphs on 40 vertices with many edges."""
import random
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n = 40
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    random.seed(42)
    
    # Bitset representation of adjacency
    # adj[u] is an integer where bit v is 1 if (u,v) is an edge
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def edge_exists(u, v):
        return (adj[u] >> v) & 1
        
    def creates_c6(u, v):
        """
        Check if adding edge (u,v) creates a C6.
        This is true if there is a simple path of length 5 from u to v.
        We use a bitset-based walk detection which is fast.
        Path: u -> a -> b -> c -> d -> v
        a in N(u), d in N(v).
        Check if there is a path of length 3 from a to d.
        """
        A = adj[u] & ~(1 << v)
        B = adj[v] & ~(1 << u)
        
        # Expand A by 1 step
        S1 = 0
        temp_A = A
        while temp_A:
            lsb = temp_A & -temp_A
            node = lsb.bit_length() - 1
            S1 |= adj[node]
            temp_A ^= lsb
        
        # Expand S1 by 1 step
        S2 = 0
        temp_S1 = S1
        while temp_S1:
            lsb = temp_S1 & -temp_S1
            node = lsb.bit_length() - 1
            S2 |= adj[node]
            temp_S1 ^= lsb
            
        # Expand S2 by 1 step
        S3 = 0
        temp_S2 = S2
        while temp_S2:
            lsb = temp_S2 & -temp_S2
            node = lsb.bit_length() - 1
            S3 |= adj[node]
            temp_S2 ^= lsb
            
        # If S3 intersects B, a walk of length 5 exists.
        # This is a necessary condition for a simple path.
        # It is not sufficient (could be non-simple walk), but for C6-free construction,
        # rejecting these is safer than accepting a C6.
        # Given the density, this heuristic is effective.
        if S3 & B:
            return True
        return False

    # 1. Fast Greedy Construction
    edges_to_try = all_edges[:]
    random.shuffle(edges_to_try)
    
    for u, v in edges_to_try:
        if not creates_c6(u, v):
            add_edge(u, v)
            
    current_edges = sum(bin(a).count("1") for a in adj) // 2
    best_edges = current_edges
    best_adj = adj[:]
    
    # 2. Simulated Annealing
    start_time = time.time()
    time_limit = 110.0
    
    T = 10.0
    cooling = 0.9995
    min_T = 0.1
    
    while T > min_T and (time.time() - start_time) < time_limit:
        for _ in range(200):
            move_type = random.random()
            
            if move_type < 0.4:
                # Try Add
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                
                if not edge_exists(u, v):
                    if not creates_c6(u, v):
                        add_edge(u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
            elif move_type < 0.8:
                # Try Swap
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if not edge_exists(u, v): continue
                
                x = random.randint(0, n-1)
                y = random.randint(0, n-1)
                if x == y: continue
                if x > y: x, y = y, x
                if edge_exists(x, y): continue
                
                remove_edge(u, v)
                if not creates_c6(x, y):
                    add_edge(x, y)
                else:
                    remove_edge(x, y)
                    add_edge(u, v)
            else:
                # Try Remove
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if edge_exists(u, v):
                    if random.random() < np.exp(-1.0 / T):
                        remove_edge(u, v)
                        current_edges -= 1
        
        T *= cooling
        
    adj = best_adj
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges

# EVOLVE-BLOCK-END
