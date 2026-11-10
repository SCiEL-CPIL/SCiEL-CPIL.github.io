import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    n = 20
    vertices = list(range(n))
    
    # Precompute all pairs for potential edges
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute triples for validation (1140 triples)
    triples = list(combinations(vertices, 3))
    
    # Helper to check if a graph (adjacency list) is K3,3-free
    def is_k33_free(adj):
        # adj is list of sets
        # For every triple (u,v,w), |N(u) ∩ N(v) ∩ N(w)| <= 2
        for u, v, w in triples:
            # Optimization: if any pair is not connected, intersection is subset of their common neighbors
            # Actually, just intersect the sets. Python set intersection is fast.
            common = adj[u] & adj[v] & adj[w]
            if len(common) > 2:
                return False
        return True
        
    # Helper to convert adjacency list to edge list
    def adj_to_edges(adj):
        edges = []
        for u in range(n):
            for v in adj[u]:
                if u < v:
                    edges.append([u, v])
        return edges
        
    # Helper to count edges
    def count_edges(adj):
        return sum(len(s) for s in adj) // 2

    # Strategy 1: Systematic Construction via BIBD(20, 4, 2)
    # A 2-(20, 4, 2) design has 20 points, block size 4, lambda=2.
    # Number of blocks b = n(n-1)lambda / (k(k-1)) = 20*19*2 / (4*3) = 760/12 = 63.33?
    # Wait. v(v-1)lambda = k(k-1)b.
    # 20*19*2 = 760. 4*3 = 12. 760/12 is not integer.
    # So a symmetric 2-(20,4,2) design does not exist?
    # Let's check parameters for v=20.
    # We need k(k-1) | v(v-1)lambda.
    # If we use a known design, e.g., from a projective plane or similar.
    # Actually, let's just use a known K3,3-free graph structure.
    # The Clebsch graph is K3,3-free? No, Clebsch is triangle-free, but K3,3 contains triangles? No, K3,3 is bipartite.
    # Wait, K3,3 is bipartite. If a graph is triangle-free, it can still contain K3,3.
    # However, the incidence graph of a 2-(v,k,lambda) design is K3,3-free if the design has no 3 blocks with 3 common points?
    # Let's use a simpler, robust method: 
    # Start with a known dense K3,3-free graph.
    # The Petersen graph is K3,3-free.
    # The complement of the line graph of K5?
    
    # Let's try a different systematic approach:
    # Randomized Greedy with Deterministic Seeds
    # Since the current program uses random restarts and gets 2.3333 (which is ~133 edges? 2.3333 * seed. Seed is likely 50? No, score is value/seed. 
    # If score is 2.3333 and seed is 50, value is 116.
    # Max edges in K3,3-free graph on 20 vertices?
    # ex(20, K3,3). 
    # Known bounds: ex(n, K3,3) ~ n^{3/2}. For n=20, 20^{1.5} = 89.
    # Actually, ex(20, K3,3) is likely around 80-100.
    # Let's look at the seed score. If seed program is basic, maybe it gets 50 edges?
    # If score 2.3333 means 116 edges, that's high.
    
    # Let's implement a faster, more aggressive local search.
    # Key improvement: Use bitmasks for adjacency for fast intersection checks.
    
    def optimize_from_seed(adj_mask, time_limit):
        # adj_mask: list of ints (bitmasks)
        # Greedy add
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not (adj_mask[i] & (1 << j)):
                        # Check if adding (i,j) violates K3,3
                        # Only triples containing i or j are affected.
                        # Precompute triples containing i or j?
                        # For speed, just check all triples. 1140 triples is small.
                        # But we can be smarter.
                        
                        # Tentatively add
                        old_i = adj_mask[i]
                        old_j = adj_mask[j]
                        adj_mask[i] |= (1 << j)
                        adj_mask[j] |= (1 << i)
                        
                        valid = True
                        # Check only triples involving i or j
                        # To avoid building lists every time, we can just loop all triples.
                        # 1140 iterations * bit_count is very fast in Python?
                        # Let's try.
                        for u, v, w in triples:
                            if i not in (u,v,w) and j not in (u,v,w):
                                continue
                            common = adj_mask[u] & adj_mask[v] & adj_mask[w]
                            if common.bit_count() > 2:
                                valid = False
                                break
                        
                        if valid:
                            changed = True
                        else:
                            adj_mask[i] = old_i
                            adj_mask[j] = old_j
                            
        # Local Search: Swaps
        # Try to swap edges to increase count
        # This is complex. Let's stick to aggressive greedy with multiple random restarts.
        
        return adj_mask

    best_edges_count = 0
    best_adj_mask = [0] * n
    
    # Use multiple random restarts with different densities
    rng = random.Random(123)
    
    # We have ~140 seconds.
    # Each greedy pass takes ~0.1s? 
    # Let's run as many as possible.
    
    for restart in range(2000):
        if time.time() > deadline:
            break
            
        # Random initialization
        density = rng.random() * 0.5 + 0.2
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < density:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    
        # Optimize
        adj = optimize_from_seed(adj, deadline)
        
        cnt = sum(m.bit_count() for m in adj) // 2
        if cnt > best_edges_count:
            best_edges_count = cnt
            best_adj_mask = list(adj)
            
        # Early exit if we hit a very high number?
        # ex(20, K3,3) is likely < 100.
        # If we get 95+, we can stop? No, keep searching.
        
    # Convert best mask to edges
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_adj_mask[i] & (1 << j):
                edges.append([i, j])
                
    return edges
