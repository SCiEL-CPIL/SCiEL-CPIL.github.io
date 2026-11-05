import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin for safety
    
    # Strategy:
    # 1. Use a deterministic 39-vertex graph derived from the icosahedron.
    #    Vertices: 12 (icosahedron vertices) + 27 (faces? no, 20 faces) + ...?
    #    Actually, a known 39-vertex graph is the complement of the "39-vertex graph"
    #    often cited in Ramsey theory.
    #    Let's construct a graph based on the icosahedron's structure.
    #    Icosahedron has 12 vertices, 30 edges, 20 faces.
    #    Total 12+20=32. Not 39.
    #    
    #    Alternative: The 39-vertex graph is often constructed from the Petersen graph or similar.
    #    Let's try a different known construction:
    #    The graph is the line graph of K_{5,5} minus a perfect matching? No, n=25.
    #    
    #    Let's use the "Icosahedron Vertex + Edge" incidence?
    #    12 vertices + 30 edges = 42 vertices.
    #    If we remove 3 vertices, we get 39.
    #    
    #    Actually, let's use a robust heuristic approach that is more likely to work:
    #    Start with a random graph on 39 vertices, but use a much smarter initialization.
    #    Initialize as a union of cliques? No, must be triangle-free.
    #    Initialize as a bipartite graph? Bipartite graphs are triangle-free.
    #    A random bipartite graph on 39 vertices (split 19/20) has alpha ~ 19+20=39? No, alpha is max independent set.
    #    In a bipartite graph, alpha = n - matching_size.
    #    To minimize alpha, we want a large matching.
    #    A random bipartite graph with p=0.5 has a perfect matching with high probability.
    #    So alpha ~ 39 - 19 = 20. Too large.
    #    
    #    We need a denser triangle-free graph.
    #    The Ramsey number R(3,10) > 39 implies there exists a 39-vertex triangle-free graph with alpha <= 9.
    #    This graph is quite dense.
    #    
    #    Let's use the "Clebsch graph" blow-up idea but more carefully.
    #    Clebsch graph: n=16, triangle-free, alpha=5.
    #    If we take a "blow-up" where each vertex is replaced by a clique? No, triangle-free.
    #    Each vertex replaced by an independent set?
    #    If we replace each vertex v of Clebsch with an independent set V_v of size k_v.
    #    Edges between V_u and V_v are complete bipartite if u~v in Clebsch, else empty.
    #    This is the "lexicographic product" or "blow-up".
    #    n = sum k_v.
    #    Triangle-free? Yes, if base is triangle-free.
    #    Alpha? An independent set in the blow-up can pick vertices from independent sets in the base.
    #    If we pick a vertex from V_u, we cannot pick from V_v if u~v.
    #    So we can pick from an independent set in the base graph.
    #    Max alpha = max_{S independent in base} sum_{v in S} k_v.
    #    Base alpha = 5.
    #    If we set all k_v = k, n = 16k.
    #    If k=2, n=32. Alpha = 5*2 = 10. Too large (we need <10).
    #    If k=1, n=16. Alpha=5.
    #    
    #    We need n=39.
    #    16k is not 39.
    #    Maybe we can use a different base graph?
    #    
    #    Let's try the "Hoffman-Singleton" graph? n=50. Too big.
    #    
    #    Let's try a local search starting from a random triangle-free graph.
    #    To make it triangle-free quickly, start with a random bipartite graph.
    #    Then add edges within partitions? No, that creates triangles if there's a path.
    #    
    #    Better initialization:
    #    Use the "Andrásfai graph" on 26 vertices?
    #    Andrásfai graph And_k is triangle-free, alpha ~ sqrt(n log n).
    #    And_1 is C5.
    #    And_2 is C7?
    #    And_3 is the 15-vertex graph?
    #    
    #    Let's use a simulated annealing or greedy local search.
    #    Objective: Minimize alpha.
    #    Constraint: No triangles.
    #    
    #    Move: Add an edge (u,v) if it doesn't create a triangle.
    #    Check: Does u and v have a common neighbor?
    #    If not, adding (u,v) is safe.
    #    Does adding (u,v) reduce alpha?
    #    It might.
    #    
    #    We will run multiple restarts.
    
    best_n = 0
    best_graph = None
    
    # Try n=40 first, then 39
    for n in [40, 39]:
        if time.time() > deadline:
            break
            
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with a random graph that is likely triangle-free
            # Start with a random bipartite graph (split n/2, n-n/2)
            # This guarantees triangle-free.
            # Then we try to add edges to reduce alpha.
            
            adj = np.zeros((n, n), dtype=bool)
            
            # Bipartition
            mid = n // 2
            # Randomly assign vertices to two sets A and B
            # For simplicity, 0..mid-1 in A, mid..n-1 in B
            # Add random edges between A and B
            for u in range(mid):
                for v in range(mid, n):
                    if random.random() < 0.5:
                        adj[u, v] = True
                        adj[v, u] = True
                        
            # Now we have a triangle-free graph.
            # We want to reduce alpha.
            # Alpha of this graph is likely large.
            # We try to add edges within A or within B?
            # No, adding an edge (u,v) where u,v in A requires that u and v have no common neighbor in B.
            # If we add an edge within A, the graph is no longer bipartite, but still triangle-free if no common neighbor.
            
            # Local Search:
            # 1. Find an independent set of size 10.
            # 2. If found, try to add an edge within this set that is "safe".
            # 3. Repeat.
            
            improved = True
            while improved and time.time() < deadline:
                improved = False
                # Find independent set of size 10
                iso = find_independent_set(adj, n, 10)
                
                if iso is None:
                    # Valid graph found
                    edges = []
                    for i in range(n):
                        for j in range(i+1, n):
                            if adj[i, j]:
                                edges.append([i, j])
                    if n > best_n:
                        best_n = n
                        best_graph = {'n': n, 'edges': edges}
                    improved = False
                    break
                else:
                    # Try to add an edge within iso
                    added = False
                    # Shuffle iso to avoid bias
                    random.shuffle(iso)
                    for i in range(10):
                        for j in range(i+1, 10):
                            u = iso[i]
                            v = iso[j]
                            if not adj[u, v]:
                                # Check if safe: no common neighbor
                                if not np.any(adj[u] & adj[v]):
                                    adj[u, v] = True
                                    adj[v, u] = True
                                    added = True
                                    improved = True
                                    break
                        if added:
                            break
                    if not added:
                        # Stuck? Try a random perturbation?
                        # Or just break and try next attempt.
                        # To escape local minima, we could remove a random edge?
                        # But removing edges increases alpha.
                        # So we are stuck.
                        break
            
            # If we didn't find a valid graph, we might try to "repair" by removing edges?
            # No, removing edges makes alpha larger.
            # So if we are stuck, this attempt failed.
            
    if best_graph is not None:
        return best_graph
    
    # Fallback: Return a known valid smaller graph
    # n=26 seed graph from current program
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

def find_independent_set(adj, n, k):
    """
    Find an independent set of size k in the graph defined by adj.
    Returns a list of vertices if found, None otherwise.
    Uses a simple backtracking search with pruning.
    """
    # Precompute non-neighbors for faster check
    non_neighbors = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if i != j and not adj[i, j]:
                mask |= (1 << j)
        non_neighbors[i] = mask
    
    best_set = []
    found = [False]
    
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
        
        # Pick a vertex from candidates
        # Heuristic: pick one with fewest non-neighbors in candidates?
        # Or just the first one.
        lsb = candidates_mask & -candidates_mask
        v = lsb.bit_length() - 1
        
        # Branch 1: Include v
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
