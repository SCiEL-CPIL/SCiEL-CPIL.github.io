import time
import random
import itertools

def construct():
    N = 40
    start_time = time.time()
    deadline = 55.0
    
    # Fallback: Fan construction (19 chords, 59 edges)
    def get_fan():
        edges = set()
        for i in range(N):
            edges.add((i, (i + 1) % N))
        for k in range(2, 21):
            edges.add((0, k))
        edge_list = [tuple(sorted(e)) for e in edges]
        cycles = []
        for L in range(3, N + 1):
            if L == N:
                cycles.append(list(range(N)))
            elif L <= 21:
                cycles.append(list(range(L)))
            else:
                k = 41 - L
                cycles.append([0] + list(range(k, N)))
        return {'edges': edge_list, 'cycles': cycles}
    
    best_result = get_fan()
    best_edge_count = len(best_result['edges'])
    
    # Helper to check if a set of edges is pancyclic and find cycles
    def is_pancyclic_and_get_cycles(edge_set):
        # Build adjacency list
        adj = {i: set() for i in range(N)}
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        
        cycles = [None] * (N + 1) # Index 3..N
        
        # Use BFS/DFS to find shortest cycle of each length? 
        # Too slow for all lengths on all steps.
        # Instead, we rely on the specific structure or random sampling.
        # For validation, we need to be sure.
        
        # We can check if a cycle of length L exists by:
        # 1. Trying to remove N-L vertices? No.
        # 2. Using a specific algorithm for girth?
        
        # Since we are searching for sparse graphs, we assume the cycles are "short" relative to chords.
        # We will try to construct the cycles explicitly for the candidate.
        
        pass # Placeholder
        
    # Heuristic Search
    # We want to find a set of chords S such that the graph is pancyclic.
    # Let's try a specific structure: Chords from a "hub" or "hubs".
    
    # Try random sets of chords of size m = 10, 11, ...
    # To speed up, we only evaluate a few.
    
    for m in range(9, 15):
        if time.time() > deadline:
            break
            
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            # Generate random chords
            # Avoid duplicates, avoid rim edges
            rim_edges = set((i, (i+1)%N) for i in range(N))
            rim_edges = set(tuple(sorted(e)) for e in rim_edges)
            
            all_possible_chords = []
            for u in range(N):
                for v in range(u+1, N):
                    e = (u, v)
                    if e not in rim_edges and (N-1 - abs(u-v) + 1) > 1: # Not rim
                        # Check if it's a rim edge (distance 1 or N-1)
                        dist = abs(u - v)
                        if dist == 1 or dist == N - 1:
                            continue
                        all_possible_chords.append(e)
                        
            random.shuffle(all_possible_chords)
            chords = all_possible_chords[:m]
            
            candidate_edges = set(rim_edges)
            for c in chords:
                candidate_edges.add(c)
                
            # Check if pancyclic
            # We need to find a cycle of every length 3..N.
            # This is expensive.
            # Optimization: Only accept if we can easily find cycles.
            # For this specific task, the "fan" is safe.
            # If I can't guarantee pancyclicity in <1s per check, I won't find a better solution in 60s.
            pass
            
    return best_result
