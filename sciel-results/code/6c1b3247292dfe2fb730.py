import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Strategy:
    # The known 39-vertex graph for R(3,10) > 39 is often constructed from the icosahedron.
    # The icosahedron has:
    # - 12 vertices (V)
    # - 30 edges (E)
    # - 20 faces (F)
    # Total entities: 12 + 30 + 20 = 62. This is too many.
    
    # A more common construction for the 39-vertex graph involves the line graph or a specific subgraph.
    # However, a well-known 39-vertex triangle-free graph with alpha=9 is the complement of the 
    # 39-vertex strongly regular graph with parameters (39, 14, 5, 6)? 
    # Actually, let's look at the "Clebsch graph" (16 vertices, alpha=5).
    # The "Hoffman-Singleton" graph (50 vertices, alpha=7).
    
    # Let's use a specific known construction for n=39.
    # One such graph is the "39-vertex Ramsey graph" which is the complement of the 
    # graph formed by the 39 edges of the complete graph K_9 minus a 1-factor? No.
    
    # Let's try a different approach: The graph is the "Andrásfai graph" A_5? No, A_5 has n=25.
    
    # Let's use the explicit construction from the literature for the 39-vertex graph.
    # It is often described as the graph whose vertices are the 39 pairs (i, j) with 0 <= i < j <= 8 
    # minus the pairs in a perfect matching of K_9? 
    # K_9 has 36 edges. 36 is not 39.
    
    # Wait, R(3,10) > 39 is a known result. The graph is usually the complement of the 
    # 39-vertex graph known as the "39-vertex Ramsey graph".
    
    # Let's use a robust local search starting from a good heuristic structure.
    # The most reliable structure for triangle-free graphs with small alpha is a "blow-up" of a 
    # triangle-free base graph, but we saw that C5 blow-ups have large alpha.
    
    # A better base: The Clebsch graph (n=16, alpha=5).
    # If we take the "lexicographic product" or "blow-up" of the Clebsch graph?
    # Blow up each vertex by k vertices.
    # n = 16 * k.
    # If k=2, n=32. Alpha = 5 * 2 = 10. (Not < 10).
    # If we can find a 39-vertex graph, it's not a simple blow-up.
    
    # Let's use the specific 39-vertex graph from the "Ramsey(3,10)" database.
    # It is the complement of the 39-vertex graph with edges defined by:
    # Vertices are 0..38.
    # This graph is the "line graph of K_{5,5} minus a perfect matching"?
    # K_{5,5} has 25 edges. Line graph has 25 vertices.
    
    # Actually, there is a known 39-vertex graph constructed from the icosahedron.
    # Let's generate the icosahedron graph and use its line graph or incidence structure.
    # Icosahedron: 12 vertices, 30 edges.
    # Line graph of Icosahedron: 30 vertices.
    
    # Let's try to construct a 39-vertex graph using a SAT/ILP-like local search but 
    # initialized with a "greedy triangle-free" structure.
    
    # However, the most efficient way to get n=39 is to use a known valid graph.
    # The graph is the complement of the "39-vertex graph" which is the 
    # "Clebsch graph" extended?
    
    # Let's use the following known valid graph for n=39:
    # It is the graph where vertices are the 39 edges of the icosahedron? No, 30 edges.
    
    # Let's use a random 39-vertex graph and a very fast local search.
    # To ensure speed, we will use bitmasks for adjacency.
    
    n_target = 39
    
    # Precompute a known valid 39-vertex graph if possible?
    # Since I don't have the explicit edge list memorized, I will use a fast local search.
    # The local search will be:
    # 1. Start with a random graph.
    # 2. Remove edges to destroy triangles (fast using bitmasks).
    # 3. Add edges to reduce alpha (fast using bitmasks).
    
    def find_independent_set_bitmask(adj_masks, n, k):
        """
        Find an independent set of size k using bitmask backtracking.
        adj_masks[i] is a bitmask of neighbors of i.
        Returns a list of vertices if found, None otherwise.
        """
        # non_neighbors[i] is a bitmask of vertices NOT connected to i (and not i)
        # non_neighbors[i] = ~adj_masks[i] & ~ (1 << i) & ((1 << n) - 1)
        
        full_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            # ~adj_masks[i] has infinite 1s in Python, so we mask it
            non_neighbors[i] = full_mask & ~adj_masks[i] & ~(1 << i)
        
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
            
            # Heuristic: pick a vertex with fewest non-neighbors in candidates?
            # Or just the first one.
            # To speed up, pick the vertex with the smallest index?
            # Let's pick the vertex with the most constraints (fewest non-neighbors in candidates)
            # to prune faster? Or least?
            # Usually, "most constrained" first is good for finding solutions or proving none exist.
            
            # For simplicity and speed, pick the first available bit.
            lsb = candidates_mask & -candidates_mask
            v = lsb.bit_length() - 1
            
            # Branch 1: Include v
            # New candidates: candidates that are non-neighbors of v
            new_candidates = candidates_mask & non_neighbors[v] & ~lsb
            dfs(new_candidates, current_set + [v])
            
            if found[0]:
                return
            
            # Branch 2: Exclude v
            dfs(candidates_mask & ~lsb, current_set)
        
        dfs(full_mask, [])
        
        if found[0]:
            return best_set
        return None

    def get_edges_from_masks(adj_masks, n):
        edges = []
        for i in range(n):
            mask = adj_masks[i]
            for j in range(i+1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    best_result = None
    best_n = 0
    
    # Try n=39
    n = 39
    full_mask = (1 << n) - 1
    
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj_masks = [0] * n
        for i in range(n):
            mask = 0
            for j in range(n):
                if i != j and random.random() < 0.5:
                    mask |= (1 << j)
            adj_masks[i] = mask
            # Ensure symmetry
            # We'll enforce symmetry by setting both bits when we modify.
            # For initialization, let's just generate upper triangle and mirror.
            
        # Better initialization:
        adj_masks = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
        
        # Phase 1: Make triangle-free
        # Repeatedly find a triangle and remove an edge.
        # A triangle (u, v, w) exists if adj[u] & adj[v] has bit w, and w > v > u.
        
        for _ in range(1000):
            if time.time() > deadline:
                break
                
            tri_found = False
            for u in range(n):
                for v in range(u+1, n):
                    if adj_masks[u] & (1 << v):
                        # Check common neighbors w > v
                        common = adj_masks[u] & adj_masks[v] & (full_mask & ~((1 << (v+1)) - 1))
                        if common:
                            # Pick one common neighbor w
                            lsb = common & -common
                            w = lsb.bit_length() - 1
                            # Remove edge (u, v)
                            adj_masks[u] &= ~(1 << v)
                            adj_masks[v] &= ~(1 << u)
                            tri_found = True
                            break
                if tri_found:
                    break
            if not tri_found:
                break
        
        # Phase 2: Reduce Alpha
        # We want alpha < 10.
        for _ in range(200):
            if time.time() > deadline:
                break
                
            # Find an independent set of size 10
            iso = find_independent_set_bitmask(adj_masks, n, 10)
            
            if iso is None:
                # Valid graph found
                edges = get_edges_from_masks(adj_masks, n)
                if n > best_n:
                    best_n = n
                    best_result = {'n': n, 'edges': edges}
                break
            else:
                # iso is a list of 10 vertices
                # Try to add an edge between two vertices in iso
                # that does not create a triangle
                improved = False
                for i in range(10):
                    for j in range(i+1, 10):
                        u = iso[i]
                        v = iso[j]
                        if not (adj_masks[u] & (1 << v)):
                            # Check if adding (u, v) creates a triangle
                            # This happens if u and v have a common neighbor
                            common = adj_masks[u] & adj_masks[v]
                            if common == 0:
                                adj_masks[u] |= (1 << v)
                                adj_masks[v] |= (1 << u)
                                improved = True
                                break
                    if improved:
                        break
                
                if not improved:
                    # Cannot improve this specific independent set easily
                    # Try to remove an edge? No, we want to add edges to reduce alpha.
                    # If we can't add an edge, we might need to remove an edge elsewhere?
                    # Or just break and try a new random graph.
                    break
    
    if best_result is not None:
        return best_result
    
    # Fallback: Return a known valid smaller graph
    # n=26 seed graph
    n = 26
    return {'n': n, 'edges': [[i, j] for i in range(n) for j in range(i + 1, n) if (j - i) % n % 3 == 1]}

import random
# Ensure random is imported at top level for the function to use
# (It was imported in the original, but let's make sure it's available)
