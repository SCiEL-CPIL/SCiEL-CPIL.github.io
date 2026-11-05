import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 55.0  # Safety margin before 60s limit
    
    N = 256
    vertices = list(range(1, N + 1))
    
    # Precompute perfect squares up to 2*N
    squares = set()
    k = 1
    while k * k <= 2 * N:
        squares.add(k * k)
        k += 1
        
    # Build conflict graph G
    # G[u] contains v if u+v is a square
    # We work with 0-indexed internally
    G = [set() for _ in range(N)]
    
    for i in range(N):
        val_i = i + 1
        for j in range(i + 1, N):
            val_j = j + 1
            if val_i + val_j in squares:
                G[i].add(j)
                G[j].add(i)
                
    # Check diagonal: 2*v is square
    valid_vertices = []
    for i in range(N):
        val_i = i + 1
        if 2 * val_i not in squares:
            valid_vertices.append(i)
            
    # Map old indices to new indices for valid vertices
    # Or just work with the subset of valid vertices
    # Let's create the complement graph H on valid_vertices
    # H[u] contains v if u and v are NOT connected in G (and u != v)
    # Since we only care about valid vertices, we can just use the list of valid vertices
    
    M = len(valid_vertices)
    # Re-index valid vertices to 0..M-1
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertices)}
    
    H = [set() for _ in range(M)]
    
    for new_i, old_i in enumerate(valid_vertices):
        for new_j, old_j in enumerate(valid_vertices):
            if new_i == new_j:
                continue
            # If old_i and old_j are NOT in G, they are connected in H
            if old_j not in G[old_i]:
                H[new_i].add(new_j)
                # H is symmetric, so we don't need to add the reverse if we iterate all pairs, 
                # but iterating all pairs i,j is O(M^2) which is fine for M~250.
                
    # Actually, building H by checking non-edges in G is O(M^2 * deg) or similar.
    # Better: Initialize H[i] = all other valid vertices, then remove edges present in G.
    # H[i] = {j in valid_vertices | j != i}
    # For each edge (u, v) in G where u, v in valid_vertices:
    #   remove v from H[u] and u from H[v]
    
    H = [set(range(M)) for _ in range(M)]
    for i in range(M):
        H[i].discard(i)
        
    for old_i in valid_vertices:
        new_i = index_map[old_i]
        for old_j in G[old_i]:
            if old_j in index_map:
                new_j = index_map[old_j]
                H[new_i].discard(new_j)
                # H is symmetric, discard other way too? 
                # If we iterate all old_i, we will hit (u,v) and (v,u).
                # So H[new_j].discard(new_i) will be handled when old_i = old_j.
                # To be safe and fast, just discard both here.
                H[new_j].discard(new_i)

    # Max Clique in H using Tomita's Algorithm
    best_clique = []
    best_size = 0
    
    # Greedy coloring for upper bound
    # We need to color the subgraph induced by P in H.
    # The number of colors is an upper bound on the clique size in that subgraph.
    def color_graph(P):
        # P is a set of vertices
        if not P:
            return 0, []
        
        # Order vertices in P by degree in H[P] descending? 
        # Tomita uses Maximum Cardinality Search order for coloring.
        # Simpler: Just greedy color in arbitrary or degree order.
        # Let's use degree in H[P] descending for better coloring quality (fewer colors).
        
        P_list = list(P)
        # Calculate degrees in induced subgraph
        degrees = {}
        for u in P_list:
            deg = len(H[u] & P)
            degrees[u] = deg
            
        # Sort by degree descending
        P_list.sort(key=lambda x: degrees[x], reverse=True)
        
        colors = {}
        color_count = 0
        # Map color to list of vertices? Or just check neighbors.
        # To check if a color is available for u, we check if any neighbor of u in P has that color.
        # Since we process in a fixed order, we only need to check neighbors processed so far?
        # No, standard greedy coloring checks all neighbors.
        # But for upper bound, we just need a valid coloring.
        
        # Efficient check:
        # For each u in P_list:
        #   used_colors = {colors[v] for v in H[u] if v in colors}
        #   assign smallest c not in used_colors
        
        for u in P_list:
            used = set()
            for v in H[u]:
                if v in colors:
                    used.add(colors[v])
            
            c = 1
            while c in used:
                c += 1
            colors[u] = c
            if c > color_count:
                color_count = c
                
        return color_count, P_list, colors

    def expand(R, P, X):
        nonlocal best_clique, best_size
        
        if not P:
            if len(R) > best_size:
                best_size = len(R)
                best_clique = list(R)
            return
            
        # Upper bound
        num_colors, P_ordered, color_map = color_graph(P)
        
        # If current size + max possible from P <= best_size, prune
        if len(R) + num_colors <= best_size:
            return
            
        # Process vertices in P in decreasing order of color
        # This is crucial for Tomita's algorithm
        P_sorted = sorted(P_ordered, key=lambda x: color_map[x], reverse=True)
        
        for v in P_sorted:
            # Check bound for this specific vertex
            # If len(R) + color_map[v] <= best_size, we can break 
            # because subsequent vertices have <= color
            if len(R) + color_map[v] <= best_size:
                break
                
            # Branch: include v
            R.add(v)
            # New P is P intersect N_H(v)
            P_new = P & H[v]
            # New X is X intersect N_H(v)
            X_new = X & H[v]
            
            expand(R, P_new, X_new)
            
            R.remove(v)
            # Move v to X
            P.remove(v)
            X.add(v)
            
            # Check time
            if time.time() - start_time > DEADLINE:
                return

    # Initial call
    R = set()
    P = set(range(M))
    X = set()
    
    # Optional: Greedy initialization to get a good lower bound quickly
    # Greedy Max Clique
    P_greedy = set(range(M))
    R_greedy = set()
    while P_greedy:
        # Pick vertex with max degree in P_greedy
        best_v = None
        max_deg = -1
        for v in P_greedy:
            deg = len(H[v] & P_greedy)
            if deg > max_deg:
                max_deg = deg
                best_v = v
        if best_v is None:
            break
        R_greedy.add(best_v)
        P_greedy = P_greedy & H[best_v]
        
    if len(R_greedy) > best_size:
        best_size = len(R_greedy)
        best_clique = list(R_greedy)
        
    expand(R, P, X)
    
    # Convert best_clique (indices in valid_vertices) back to original values
    result = []
    for idx in best_clique:
        original_idx = valid_vertices[idx]
        result.append(original_idx + 1)
        
    # Verify
    valid = True
    res_set = set(result)
    for a in res_set:
        if 2*a in squares:
            valid = False
            break
        for b in res_set:
            if a <= b and a + b in squares:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        return list(range(1, 257, 3))
        
    return result
