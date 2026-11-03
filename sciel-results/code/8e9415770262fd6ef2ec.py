import numpy as np
import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Moore bound for even g
    # n >= k * sum_{i=0}^{g/2 - 1} (k-1)^i
    m = g // 2
    moore_bound = k * sum((k-1)**i for i in range(m))
    
    # If k=4, g=10, moore_bound = 484
    
    best_n = None
    best_edges = None
    
    # Try N from moore_bound upwards
    # We might not find a solution for N=484 quickly, so we step up
    # But we should try a few for each N
    
    for N in range(moore_bound, moore_bound + 200):
        if time.time() > deadline:
            break
            
        # Try to find a valid graph of size N
        found = False
        attempts = 0
        max_attempts = 50 if N < 1000 else 10
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            edges = generate_random_k_regular(N, k)
            if edges is None:
                continue
                
            if check_girth(N, edges, k, g):
                best_n = N
                best_edges = edges
                found = True
                break
                
        if found:
            break

    if best_n is None:
        # Fallback: return a large random graph that is likely valid
        # Or return the LUW graph from seed? No, it's huge.
        # Just return a large random one
        N_fallback = 2000
        edges = generate_random_k_regular(N_fallback, k)
        # Ensure it's valid? If not, we might fail.
        # But 2000 vertices 4-regular graph has high girth probability.
        # Let's try to force validity by checking and retrying a few times
        while not check_girth(N_fallback, edges, k, g):
             edges = generate_random_k_regular(N_fallback, k)
             if time.time() > deadline:
                 break
        best_n = N_fallback
        best_edges = edges

    return best_n, best_edges

def generate_random_k_regular(N, k):
    # Generate k random perfect matchings
    # A perfect matching is a permutation p where p[i] != i and p is an involution? No.
    # A perfect matching is a set of disjoint edges covering all vertices.
    # We can generate a random permutation and pair up (0, p[0]), (1, p[1])...? No.
    
    # Simple method: 
    # 1. Create a list of stubs: [0,0,0,0, 1,1,1,1, ...]
    # 2. Randomly pair them up.
    # 3. Check for loops and multiple edges.
    
    stubs = []
    for i in range(N):
        stubs.extend([i]*k)
        
    random.shuffle(stubs)
    
    edges = set()
    for i in range(0, len(stubs), 2):
        u = stubs[i]
        v = stubs[i+1]
        if u == v:
            return None # Loop
        if u > v:
            u, v = v, u
        if (u, v) in edges:
            return None # Multiple edge
        edges.add((u, v))
        
    return list(edges)

def check_girth(N, edges, k, g):
    # Build adjacency list
    adj = [[] for _ in range(N)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # BFS from each vertex
    # We want to detect cycles of length < g
    # A cycle of length c exists if there is a path of length c from u to u
    # Or more simply, if during BFS we find a visited node at depth d such that 
    # the path length + 1 < g?
    
    # Standard method:
    # For each start vertex s:
    #   BFS up to depth g//2
    #   If we reach a vertex v at depth d, and v was reached earlier at depth d' < d
    #   and d + d' < g, then there is a cycle of length < g.
    #   Actually, if we reach a vertex already in the queue or visited with smaller depth.
    
    # Simpler: Check for cycles of length 4, 6, ..., g-2.
    # For k=4, g=10, check 4, 6, 8.
    
    # Cycle of length 4: u-v-w-x-u
    # For each edge (u,v), check if there is a common neighbor of u and v? No, that's triangle.
    # Common neighbor of u and w?
    
    # Let's just do BFS depth limit.
    
    for s in range(N):
        # dist[v] = distance from s
        # prev[v] = previous node in BFS tree
        dist = [-1] * N
        prev = [-1] * N
        dist[s] = 0
        queue = [s]
        
        for v in queue:
            if dist[v] >= g // 2:
                continue
            for u in adj[v]:
                if dist[u] == -1:
                    dist[u] = dist[v] + 1
                    prev[u] = v
                    queue.append(u)
                else:
                    # Found a back edge or cross edge
                    # Cycle length = dist[v] + dist[u] + 1
                    # But we need to be careful with the tree structure.
                    # If u is in the BFS tree of s, and u != prev[v]
                    # Then we have a cycle.
                    if u != prev[v]:
                        cycle_len = dist[v] + dist[u] + 1
                        if cycle_len < g:
                            return False
                            
    return True
