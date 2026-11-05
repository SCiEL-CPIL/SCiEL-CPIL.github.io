import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # The strategy is to use a deterministic construction that is known to be triangle-free
    # and has small independence number, then verify/repair if necessary.
    
    # The "Polarity Graph" over GF(q) for prime q.
    # Vertices: (x, y) in GF(q) x GF(q). Total n = q^2.
    # Edge between (x1, y1) and (x2, y2) if x1*x2 + y1*y2 = 1.
    # This graph is triangle-free.
    # Its independence number alpha is roughly q * sqrt(q) / log(q) or similar bounds.
    # For q=5, n=25. alpha is small.
    # For q=7, n=49. Too big? R(3,10) <= 41. So n must be <= 40.
    # So we can't use q=7 directly (n=49).
    # We can use q=5 (n=25) and try to extend? Or use a different structure.
    
    # Alternative: The "Andrásfai Graph" And_k.
    # And_3 is the 15-vertex graph? No, And_k has n=2k+1?
    # Actually, the known 39-vertex graph is often constructed from the icosahedron or similar.
    # But random search found 38-40. Let's use a smarter random search.
    
    # New Strategy:
    # 1. Start with a "dense" triangle-free graph.
    #    The complete bipartite graph K_{n/2, n/2} is triangle-free but alpha = n/2.
    #    We need alpha < 10. So n/2 < 10 => n < 20. Not enough.
    #    We need to add edges within the partitions to reduce alpha, while maintaining triangle-free.
    
    #    Better initialization:
    #    Use a "Blow-up" of a small triangle-free graph with small alpha.
    #    Example: C5 (n=5, alpha=2).
    #    Blow-up each vertex by size k. n = 5k.
    #    If k=8, n=40.
    #    Alpha of blow-up = k * alpha(base) = 8 * 2 = 16. Too large.
    #    We need alpha < 10.
    #    So k * 2 < 10 => k < 5. n < 25.
    
    #    We need a base graph with smaller alpha.
    #    Clebsch graph: n=16, alpha=5.
    #    Blow-up by k. n = 16k.
    #    Alpha = 5k.
    #    If k=2, n=32, alpha=10. (Need <10, so alpha<=9).
    #    If k=1, n=16, alpha=5.
    #    If we use k=2, we get n=32, alpha=10. We need to break the alpha=10 sets.
    
    #    Let's try to construct a graph on n=39 or 40 using a hybrid approach.
    #    1. Start with a random graph.
    #    2. Remove all triangles (greedy or random order).
    #    3. This might leave many independent sets.
    #    4. Add edges to break independent sets.
    
    #    Actually, the "random greedy" that scored 1.192 (n=38-40) is a good baseline.
    #    Let's improve it.
    
    #    Improved Random Greedy:
    #    1. Initialize with empty graph on n=40.
    #    2. Randomly order all possible edges.
    #    3. Add edge if it doesn't create a triangle.
    #    4. This yields a maximal triangle-free graph.
    #    5. Check alpha. If alpha >= 10, try to repair.
    #       Repair: Find an independent set of size 10.
    #       Try to add an edge within this set that doesn't create a triangle.
    #       If possible, add it.
    #       Repeat until no independent set of size 10 exists or stuck.
    
    #    To increase success rate:
    #    - Use multiple restarts.
    #    - Use a "biased" random order for edges? Or just random.
    #    - Use a better independent set finder (backtracking with pruning).
    
    best_n = 0
    best_edges = []
    
    # Try n=40, then 39, then 38...
    for n in [40, 39, 38, 37]:
        if time.time() > deadline:
            break
            
        for attempt in range(20):
            if time.time() > deadline:
                break
            
            adj = np.zeros((n, n), dtype=bool)
            
            # Step 1: Random Greedy Triangle-Free Construction
            # Generate random permutation of all edges
            edges_list = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(edges_list)
            
            for u, v in edges_list:
                # Check if adding (u,v) creates a triangle
                # i.e., is there w such that adj[u,w] and adj[v,w]?
                if not np.any(adj[u] & adj[v]):
                    adj[u, v] = True
                    adj[v, u] = True
            
            # Step 2: Repair Phase
            # Find independent set of size 10.
            # If found, try to add an edge within it.
            improved = True
            while improved and time.time() < deadline:
                improved = False
                iso = find_independent_set(adj, n, 10)
                
                if iso is None:
                    # Valid!
                    if n > best_n:
                        best_n = n
                        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i, j]]
                    break
                else:
                    # Try to add an edge within iso
                    added = False
                    # Shuffle to avoid bias
                    random.shuffle(iso)
                    for i in range(10):
                        for j in range(i+1, 10):
                            u = iso[i]
                            v = iso[j]
                            if not adj[u, v]:
                                if not np.any(adj[u] & adj[v]):
                                    adj[u, v] = True
                                    adj[v, u] = True
                                    added = True
                                    improved = True
                                    break
                        if added:
                            break
                    if not added:
                        # Stuck. Try a perturbation?
                        # Remove a random edge? No, that increases alpha.
                        # Maybe try a different random seed for the initial greedy?
                        # We are in the attempt loop, so next attempt will restart.
                        break
            
            # If we broke out because we found a valid graph, we can stop this n
            if best_n == n:
                break
        
        # If we found a valid graph for this n, we can stop trying smaller n
        # because we want to maximize n.
        if best_n == n:
            break
    
    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    
    # Fallback: Return a known valid smaller graph
    # n=26 seed graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def find_independent_set(adj, n, k):
    """
    Find an independent set of size k in the graph defined by adj.
    Returns a list of vertices if found, None otherwise.
    Uses backtracking with pruning.
    """
    # Precompute non-neighbors as bitmasks for speed
    non_neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i, j]:
                mask |= (1 << j)
        non_neighbors[i] = mask
    
    found = [False]
    best_set = []
    
    def dfs(candidates_mask, current_set):
        if found[0]:
            return
        if len(current_set) == k:
            best_set.extend(current_set)
            found[0] = True
            return
        
        # Pruning: if not enough candidates left
        if bin(candidates_mask).count('1') < k - len(current_set):
            return
        
        # Heuristic: Pick a vertex from candidates.
        # To speed up, we can pick a vertex that has the fewest non-neighbors in the candidate set?
        # Or just the first one.
        # Let's try picking the vertex with the smallest degree in the candidate subgraph (induced by non-neighbors).
        # Actually, for independent set, we want to branch on a vertex that constrains the most.
        # Let's just take the first set bit for simplicity, but maybe order by degree?
        
        # Simple heuristic: take the lowest index vertex in candidates.
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
        # New candidates must be non-neighbors of v
        new_candidates = candidates_mask & non_neighbors[v] & ~lsb
        dfs(new_candidates, current_set + [v])
        
        if found[0]:
            return
        
        # Branch 2: Exclude v
        dfs(candidates_mask & ~lsb, current_set)
    
    dfs((1 << n) - 1, [])
    
    if found[0]:
        return best_set
    return None
