import numpy as np
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Fallback: Paley(17)
    def paley_17():
        squares = {x * x % 17 for x in range(1, 17)}
        return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    
    best_result = None
    best_score = -1
    
    # Try to find a graph with n=35
    # We will use Simulated Annealing on the edge set
    n = 35
    
    # Helper to check validity
    def is_valid(G):
        # G is a numpy array of 0/1
        # Check K4
        for u in range(n):
            for v in range(u+1, n):
                if G[u, v]:
                    # Find common neighbors
                    common = np.where(G[u] & G[v] & ~np.eye(n, dtype=bool))[0]
                    # Check if any pair in common is connected
                    for i in range(len(common)):
                        for j in range(i+1, len(common)):
                            if G[common[i], common[j]]:
                                return False
        # Check I6
        # This is slow, so we only do it if K4 check passes?
        # Actually, we need to check both.
        # For I6, we can check all 6-sets.
        # This is too slow for every step in SA.
        # So we only use is_valid at the end of a phase or when penalty is 0.
        # For SA, we use a penalty function.
        pass
        
    # Penalty Function
    # We maintain K4_count and I6_count
    # But updating I6_count is hard.
    
    # Alternative: Random Greedy Construction
    # Start with empty graph. Add edges.
    # If adding edge creates K4, don't add it (or remove others).
    # If we have I6, remove edges inside it.
    
    # Let's try a simple Local Search with a fast "Almost Valid" check.
    # We'll use a random graph and flip edges.
    # We'll estimate penalty by sampling? No, must be exact.
    
    # Let's try to construct a graph based on the **Petersen Graph** structure.
    # The 35-vertex graph is often related to the **Line Graph** of K_7?
    # K_7 has 21 edges. Line graph L(K_7) has 21 vertices.
    # L(K_7) is K_4-free?
    # Vertices of L(K_7) are edges of K_7.
    # Adjacency if edges share a vertex.
    # Clique number of L(K_7) is 3 (star K_{1,3} in K_7 -> triangle in L(K_7)).
    # So L(K_7) is K_4-free.
    # Independence number of L(K_7)?
    # Independent set in L(G) is a matching in G.
    # Max matching in K_7 is floor(7/2) = 3.
    # So alpha(L(K_7)) = 3.
    # This is a graph on 21 vertices with omega=3, alpha=3.
    # We need omega <= 5 and alpha <= 5.
    # L(K_7) is too small.
    
    # What about L(K_8)?
    # Vertices: 28.
    # Omega(L(K_8)) = 3 (from K_{1,3} in K_8? No, max degree in K_8 is 7. Star K_{1,7} gives clique of size 7 in L(K_8)?
    # Wait. In L(G), a clique corresponds to a set of edges in G that are pairwise adjacent.
    # This is either a star (all edges share a vertex) or a triangle.
    # In K_8, a star has 7 edges. So L(K_8) has a K_7.
    # So L(K_8) is NOT K_4-free.
    
    # We need a graph G such that L(G) is K_4-free.
    # This requires max degree of G <= 3?
    # If max degree <= 3, then max clique in L(G) is 3 (from star K_{1,3}).
    # Also need triangle in G to avoid larger cliques?
    # If G is triangle-free and max degree <= 3, then L(G) is K_4-free.
    # Max matching in G gives alpha(L(G)).
    # We want alpha(L(G)) <= 5.
    # So max matching <= 5.
    # We want n = |E(G)| to be large.
    # If G is a union of triangles?
    # If G is a collection of disjoint triangles, max degree 2.
    # L(G) is disjoint union of K_3's.
    # K_4-free.
    # Alpha(L(G)) = number of triangles (since we can pick one edge from each triangle).
    # To have alpha <= 5, we can have at most 5 triangles.
    # n = 5 * 3 = 15.
    
    # If G is not triangle-free, we need to be careful.
    
    # Let's go back to **Simulated Annealing** on n=35.
    # It is the most likely to succeed if implemented efficiently enough.
    
    # Efficient SA:
    # 1. Random Init.
    # 2. Calculate initial K4 and I6 counts.
    # 3. Loop:
    #    - Pick random edge (u,v).
    #    - Calculate delta K4 and delta I6.
    #    - Accept if delta < 0 or with prob exp(-delta/T).
    
    # Delta K4 is fast.
    # Delta I6 is the bottleneck.
    # Approximation: If we add edge (u,v), we destroy I6s containing u,v.
    # Count of I6s containing u,v is the number of independent sets of size 4 in W_uv.
    # If we assume the graph is "random", we can estimate this?
    # No, we need exactness for the final check.
    # But for SA, we can use an **Approximate** penalty to guide the search,
    # and then **Verify** exactly when we think we are close.
    
    # Strategy:
    # Use a fast approximate penalty (e.g., count of "almost" K4/I6).
    # Or just use K4 count as primary, and check I6 periodically.
    
    # Let's try: Minimize K4 count.
    # If K4 count is 0, check I6.
    # If I6 > 0, try to break I6s by adding edges inside them.
    
    # This is a 2-phase approach.
    
    # Phase 1: Make it K4-free.
    # Phase 2: Make it I6-free.
    
    # Phase 1:
    # Start with random graph.
    # While K4 > 0:
    #   Find a K4.
    #   Remove a random edge from it.
    #   Repeat.
    # This might create I6s.
    
    # Phase 2:
    # While I6 > 0:
    #   Find an I6.
    #   Add a random edge inside it.
    #   Repeat.
    # This might create K4s.
    
    # So we iterate Phase 1 and Phase 2.
    
    # This is a valid local search.
    
    def count_k4(G):
        count = 0
        # G is list of sets of neighbors
        for u in range(n):
            for v in range(u+1, n):
                if v in G[u]:
                    common = G[u] & G[v]
                    # Count edges in common
                    for w in common:
                        if w > v: # avoid double counting?
                            # Check if w is connected to any other in common?
                            # No, K4 is u,v,w,x.
                            # We need pairs (w,x) in common that are connected.
                            pass
        # This is O(n^5) roughly.
        # For n=35, 35^5 = 52M. Too slow in Python.
        
    # We need a faster K4 count.
    # Iterate over all 4-sets? 52k sets.
    # For each set, check 6 edges.
    # 52k * 6 = 312k operations. This is fast!
    
    def has_k4(G):
        # G is adjacency matrix
        for i in range(n):
            for j in range(i+1, n):
                if G[i, j]:
                    for k in range(j+1, n):
                        if G[i, k] and G[j, k]:
                            for l in range(k+1, n):
                                if G[i, l] and G[j, l] and G[k, l]:
                                    return True
        return False
        
    def has_i6(G):
        # G is adjacency matrix. Independent set means NO edges.
        for i in range(n):
            for j in range(i+1, n):
                if not G[i, j]:
                    for k in range(j+1, n):
                        if not G[i, k] and not G[j, k]:
                            for l in range(k+1, n):
                                if not G[i, l] and not G[j, l] and not G[k, l]:
                                    for m in range(l+1, n):
                                        if not G[i, m] and not G[j, m] and not G[k, m] and not G[l, m]:
                                            return True
        return False
        
    # These are O(n^4) and O(n^6).
    # n=35.
    # K4: 35^4 / 24 * 6 checks?
    # The loops above are pruned.
    # Worst case: Dense graph.
    # I6: Sparse graph.
    
    # Let's run this.
    
    G = np.random.rand(n, n) > 0.5
    np.fill_diagonal(G, 0)
    G = (G + G.T) // 2
    
    # Symmetrize
    G = np.triu(G, 1)
    G = G + G.T
    
    for step in range(100000):
        if time.time() > deadline:
            break
            
        # Check if valid
        if not has_k4(G) and not has_i6(G):
            # Convert to edge list
            edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if G[i, j]:
                        edges.append([i, j])
            return {'n': n, 'edges': edges}
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v: continue
        
        # Flip
        G[u, v] = 1 - G[u, v]
        G[v, u] = G[u, v]
        
        # Simple acceptance: Always accept? (Random Walk)
        # Or accept if it reduces "badness"?
        # Let's just do Random Walk with occasional restarts.
        
    # If we didn't find it, try a different initialization or return fallback
    return paley_17()

# EVOLVE-BLOCK-END
