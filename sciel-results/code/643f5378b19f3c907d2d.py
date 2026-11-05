import time
import random

def construct():
    n = 16
    
    # Generate Clebsch Graph (16 vertices, 40 edges, triangle-free, 4-chromatic)
    # Vertices are 0..15 (binary 0000 to 1111)
    # Edge if Hamming distance is 1 or 4
    def hamming_dist(u, v):
        return bin(u ^ v).count('1')
        
    clebsch_edges = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_dist(i, j)
            if d == 1 or d == 4:
                clebsch_edges.add((i, j))
                
    # Initial best is Clebsch
    best_edges = list(clebsch_edges)
    best_len = len(best_edges)
    
    # All possible edges not in Clebsch
    all_non_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in clebsch_edges:
                all_non_edges.append((i, j))
                
    start_time = time.time()
    max_time = 110.0 # Leave 10s buffer
    
    # Greedy Randomized Search
    # We can also use Simulated Annealing, but Greedy with many restarts is often sufficient for this size
    # Let's use a simple local search: Add edges greedily, then try to swap?
    # Since adding edges preserves 4-chromaticity, we just need to maximize edges subject to triangle-free.
    
    # Strategy: 
    # 1. Start with Clebsch.
    # 2. Randomly shuffle non-edges.
    # 3. Add them if no triangle created.
    # 4. This is a maximal triangle-free supergraph of Clebsch.
    # 5. Repeat many times.
    
    while time.time() - start_time < max_time:
        current_edges = set(clebsch_edges)
        adj = [set() for _ in range(n)]
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        candidates = all_non_edges[:]
        random.shuffle(candidates)
        
        for u, v in candidates:
            if not (adj[u] & adj[v]):
                current_edges.add((u, v))
                adj[u].add(v)
                adj[v].add(u)
                
        if len(current_edges) > best_len:
            best_len = len(current_edges)
            best_edges = list(current_edges)
            
    return [[u, v] for u, v in best_edges]
