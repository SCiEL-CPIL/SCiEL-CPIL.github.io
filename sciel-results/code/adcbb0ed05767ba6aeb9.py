import random
import numpy as np
import time

def construct():
    start_time = time.time()
    time_limit = 110  # Leave buffer for overhead
    
    # Try to find a graph for n=35, then 34, etc.
    for n in range(35, 16, -1):
        # Multiple restarts for each n
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
                
            # Initialize random graph
            # p=0.5 is a good start for Ramsey graphs
            adj = (np.random.rand(n, n) < 0.5).astype(np.int8)
            np.fill_diagonal(adj, 0)
            # Symmetrize
            adj = np.triu(adj, 1)
            adj = adj + adj.T
            
            # Local Search
            for step in range(10000):
                if time.time() - start_time > time_limit:
                    break
                    
                # Check for K4
                k4 = find_k4(adj)
                if k4:
                    # Pick random edge in K4 to remove
                    # Edges: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
                    edges_in_k4 = [(k4[0], k4[1]), (k4[0], k4[2]), (k4[0], k4[3]),
                                   (k4[1], k4[2]), (k4[1], k4[3]), (k4[2], k4[3])]
                    u, v = random.choice(edges_in_k4)
                    adj[u, v] = 0
                    adj[v, u] = 0
                    continue
                
                # Check for Ind6
                ind6 = find_ind6(adj)
                if ind6:
                    # Pick random pair in Ind6 to add edge
                    # Pairs: all C(6,2)
                    pairs = [(ind6[i], ind6[j]) for i in range(6) for j in range(i+1, 6)]
                    u, v = random.choice(pairs)
                    adj[u, v] = 1
                    adj[v, u] = 1
                    continue
                
                # No violations found
                # Verify exactly
                if not find_k4(adj) and not find_ind6(adj):
                    # Convert to edge list
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    return {'n': n, 'edges': edges}
            
    # Fallback: Return the best known small graph (Paley 17) if all else fails
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}

def find_k4(adj):
    n = adj.shape[0]
    # Optimize: Precompute neighbor lists?
    # For n=35, simple loop is fine.
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j]:
                # Common neighbors
                # Using numpy is fast for intersection
                common = np.where(adj[i] & adj[j])[0]
                # Check edges within common
                # If len(common) < 2, no K4 possible with this pair
                if len(common) < 2:
                    continue
                # Check if any edge exists in common
                # We can just check all pairs in common
                # To speed up, we can check if the subgraph induced by common has any edge
                # Since common size is small on average, loop is ok.
                for k_idx in range(len(common)):
                    for l_idx in range(k_idx+1, len(common)):
                        k = common[k_idx]
                        l = common[l_idx]
                        if adj[k, l]:
                            return [i, j, k, l]
    return None

def find_ind6(adj):
    n = adj.shape[0]
    # Find independent set of size 6
    # Use backtracking
    # To make it fast, we can use a greedy heuristic to order vertices
    
    # Let's use a simple recursive function
    def dfs(candidates, current_set):
        if len(current_set) == 6:
            return current_set[:]
        if len(candidates) + len(current_set) < 6:
            return None
        
        # Heuristic: Sort candidates by degree in the induced subgraph of candidates?
        # This is expensive. Let's just pick the vertex with the fewest neighbors in the *remaining* candidates?
        # Or just pick the first one.
        
        # To improve performance, we can try vertices in order of increasing degree in the whole graph?
        # No, dynamic degree is better.
        
        # Let's just iterate.
        # Optimization: If a vertex is connected to ALL other candidates, it can't be in an independent set with them.
        # But we are building the set, so we remove neighbors.
        
        for idx, v in enumerate(candidates):
            # New candidates are those in candidates[idx+1:] that are NOT neighbors of v
            # This ensures we don't pick the same vertex twice and maintain order
            new_candidates = []
            # Use list comprehension for speed
            # adj[v] is a row. 
            # We need w in candidates[idx+1:] such that adj[v, w] == 0
            # 
            # This loop is the bottleneck.
            # For n=35, it's small.
            for w in candidates[idx+1:]:
                if not adj[v, w]:
                    new_candidates.append(w)
            
            res = dfs(new_candidates, current_set + [v])
            if res:
                return res
        return None

    # Initial candidates: all vertices
    # We can prune initial candidates: if a vertex has degree > n-6, it might still be in Ind6, but...
    # Actually, just start with all.
    return dfs(list(range(n)), [])
