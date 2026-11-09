import time

def construct():
    start_time = time.time()
    deadline = start_time + 88.0  # Safety margin
    
    n = 256
    
    # Build conflict graph
    # adj[i] is a bitmask of nodes that CONFLICT with i (i.e., (a*b)%(a+b)==0)
    # We want an Independent Set in this conflict graph.
    # This is equivalent to a Maximum Clique in the complement graph.
    # Let's work directly with the conflict graph and find the MIS.
    # Or, build the complement adjacency for the clique solver.
    
    # Let's use the conflict graph and find MIS using a B&B algorithm 
    # optimized for bitsets, similar to the MCS algorithm for Max Clique.
    
    # adj[i] contains bits for j such that i and j conflict.
    adj = [0] * (n + 1)
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
    
    # For the MIS problem, we want to select a set S such that no two vertices in S are adjacent in adj.
    # We can use a branch and bound algorithm.
    # State: 
    #   P: set of candidates that can be added (bitmask)
    #   R: set of currently selected vertices (bitmask)
    #   X: set of vertices that were excluded (not strictly needed for MIS B&B if we order properly, 
    #      but in MCS for clique, X is used to avoid re-considering. For MIS, we just remove from P).
    
    # Let's implement a recursive B&B for MIS.
    # To get a good upper bound, we can use the fact that the MIS size is bounded by the size of a 
    # maximum matching in the induced subgraph? No, that's for bipartite.
    # For general graphs, the greedy coloring of the *complement* of the induced subgraph on P gives an upper bound?
    # Actually, the standard MIS B&B uses the coloring of the *induced subgraph* on P to bound the clique size 
    # in the complement, which is the MIS size in the original.
    # Wait. 
    # MIS in G = Max Clique in \bar{G}.
    # The MCS algorithm finds Max Clique in a graph H.
    # If we set H = \bar{G}, then we find the Max Clique in \bar{G}, which is the MIS in G.
    # The coloring in MCS is applied to H (the complement).
    
    # So, let's build the complement adjacency matrix `comp_adj`.
    # comp_adj[i] has bit j set if i and j do NOT conflict (and i != j).
    # Note: We only care about i < j or i > j? Bitmasks handle both.
    
    comp_adj = [0] * (n + 1)
    for i in range(1, n + 1):
        # All nodes j != i
        all_others = ((1 << (n + 1)) - 2) ^ (1 << i)
        # Remove conflicts
        comp_adj[i] = all_others & ~adj[i]
        
    # MCS Algorithm for Max Clique in comp_adj
    # R: current clique (bitmask)
    # P: candidates (bitmask)
    # X: excluded (bitmask) - used to prevent re-visiting in some variants, 
    #    but in the standard Tomita algorithm, we just order P and process.
    # The key is the coloring of P to get an upper bound.
    
    best_clique_size = 0
    best_clique_mask = 0
    
    def color_sort(P):
        """
        Greedy coloring of the subgraph induced by P in comp_adj.
        Returns a list of vertices in P, ordered by their color (ascending).
        The color of a vertex is an upper bound on the clique size in the subgraph 
        induced by that vertex and all vertices with lower or equal color? 
        Actually, in MCS, we color P. The color `c` of a vertex `u` means that 
        the maximum clique in the subgraph induced by {u} U {v in P | color(v) <= color(u)} 
        is at most `c`? No.
        
        Standard MCS Coloring:
        We assign colors 1, 2, ... to vertices in P.
        The vertices are ordered by color.
        The bound for a vertex at position k (0-indexed) in the sorted list is 
        the number of distinct colors used up to that point?
        Actually, the bound is simply the color number assigned to the last vertex 
        if we process them in reverse order of color.
        
        Let's use the standard implementation:
        1. Greedily color the graph induced by P.
        2. Sort vertices by color.
        3. Iterate through the sorted list in reverse (highest color first).
        4. If |R| + color_index <= best_size, prune.
        """
        if P == 0:
            return []
        
        uncolored = P
        colors = {}
        color_count = 0
        # We will store the color for each vertex in a dict or array
        # To be fast, we can just return the list of (vertex, color)
        
        # Greedy coloring
        # Order vertices by degree in P (descending) often helps
        # But simple greedy is okay for n=256
        
        while uncolored:
            color_count += 1
            # Find a maximal independent set in the remaining uncolored graph?
            # No, greedy coloring picks a vertex, assigns current color, removes it and its neighbors from consideration for this color.
            
            # To make it deterministic and fast, we can just pick the first available vertex?
            # Better: pick vertex with highest degree in uncolored?
            
            # Let's just iterate through uncolored bits.
            # For efficiency, we'll just assign colors greedily.
            
            # We need to know which vertices have already been colored in this pass.
            # This is a standard greedy coloring algorithm.
            
            # Optimization: 
            # The bound is the number of colors.
            # We want to maximize the bound? No, we want a *tight* bound (small).
            # Greedy coloring usually gives a good enough bound.
            
            # Let's implement a simple greedy coloring that returns the list of vertices 
            # sorted by the order they were assigned colors (1, 1, 1, 2, 2, 3...).
            # Actually, the standard MCS sorts by color, and within same color, by degree or index.
            
            pass # Placeholder, will implement below

    # Let's implement a robust MCS step by step.
    
    def expand(R, P, X):
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() > deadline:
            return
            
        # If P is empty, update best
        if P == 0:
            r_size = R.bit_count()
            if r_size > best_clique_size:
                best_clique_size = r_size
                best_clique_mask = R
            return
            
        # Color the vertices in P to get an upper bound
        # We need to sort P by color.
        # color[u] is the color assigned to u.
        # The upper bound for a vertex u in the sorted list (processed in reverse) 
        # is the number of colors used up to u? 
        # In Tomita's algorithm, the vertices are sorted by color.
        # Let sorted_P be the list of vertices in P sorted by their color.
        # We process them in reverse order (highest color first).
        # The bound for the i-th vertex (0-indexed from the end) is 
        # (total_colors - i)? No.
        
        # Correct MCS Logic:
        # 1. Color P. Let the colors be c_1, c_2, ..., c_m.
        # 2. Sort vertices in P by color: v_1, v_2, ..., v_k such that color(v_1) <= color(v_2) <= ...
        # 3. For i from k down to 1:
        #    - Let u = v_i.
        #    - If |R| + i <= best_clique_size: break. (Because the max clique in {v_1...v_i} is at most i? 
        #      No, the max clique size in a graph with max degree constraints... 
        #      Actually, the property is: The max clique in the subgraph induced by {v_1, ..., v_i} 
        #      is at most i? No, that's not right.
        
        # Let's look up the exact bound in MCS.
        # The coloring assigns colors 1..k.
        # The vertices are ordered such that all vertices of color 1 come first, then 2, etc.
        # The bound for vertex v_i (the i-th in the sorted list, 1-indexed) is the number of 
        # distinct colors present in {v_1, ..., v_i}. 
        # Wait, if we have colors 1, 1, 1, 2, 2, 3.
        # v_1 (color 1): bound 1?
        # v_3 (color 1): bound 1?
        # v_4 (color 2): bound 2?
        # v_6 (color 3): bound 3?
        
        # Actually, the standard implementation uses the *index* in the sorted list as the bound?
        # No, that would be very loose.
        
        # Correct Interpretation:
        # The greedy coloring partitions P into independent sets C_1, C_2, ..., C_k.
        # Any clique in P can contain at most one vertex from each C_i.
        # Therefore, the size of the maximum clique in P is at most k.
        # When we branch on a vertex u, we consider the subgraph induced by P ∩ N(u).
        # The vertices in P ∩ N(u) are a subset of P.
        # The coloring of P gives an upper bound k for the clique size in P.
        # But we process vertices in a specific order.
        
        # Let's use the "color number" as the bound for the *last* vertex in the color class?
        # No, let's just calculate the coloring and store the color for each vertex.
        # Then sort vertices by color.
        # When we pick a vertex u from the end of the sorted list, 
        # the number of colors used by the remaining vertices (those before u in the list) 
        # is an upper bound on the clique size we can still find.
        
        # Let's implement `get_color_bound(P)` which returns a list of (vertex, color) sorted by color.
        
        def get_color_bound(P):
            if P == 0:
                return []
            
            uncolored = P
            colors = {}
            current_color = 0
            
            # To make the coloring effective, we should process vertices in a specific order.
            # Usually, highest degree first.
            
            # For simplicity and speed, let's just iterate.
            # But we need to assign colors greedily.
            
            # This standard greedy coloring is O(N^2) which is fine for N=256.
            
            # We will store the color for each vertex in a dictionary.
            
            while uncolored:
                current_color += 1
                # Find a vertex in uncolored to assign this color
                # Then remove it and its neighbors from uncolored for this color class?
                # No, in greedy coloring, we pick a vertex, assign color, and remove it from uncolored.
                # Its neighbors CANNOT have this color, but they remain in uncolored for future colors.
                # Wait, that's not how greedy coloring works for bounding cliques.
                
                # For the upper bound of a clique, we need a *proper* coloring of the graph.
                # In a proper coloring, adjacent vertices have different colors.
                # A clique can have at most one vertex of each color.
                # So the number of colors is an upper bound on the clique size.
                
                # Greedy Coloring Algorithm:
                # 1. Order vertices (e.g., by degree).
                # 2. For each vertex in order, assign the smallest available color 
                #    not used by its already-colored neighbors.
                
                # Let's do that.
                
                pass # Will implement in main function for clarity

        # Let's restructure to use a global helper for coloring.
        
        return

    # --- Revised Plan ---
    # 1. Helper: `greedy_color(P)` returns list of (vertex, color) sorted by color.
    # 2. Helper: `expand(R, P)` implements MCS.
    
    def greedy_color(P):
        """
        Returns a list of tuples (vertex, color) for vertices in P, sorted by color.
        The color is a proper coloring of the subgraph induced by P in comp_adj.
        """
        if P == 0:
            return []
        
        # Get list of vertices
        vertices = []
        t = P
        while t:
            lsb = t & (-t)
            v = lsb.bit_length() - 1
            vertices.append(v)
            t &= t - 1
            
        # Sort by degree in P (descending) to improve coloring quality
        def deg_in_P(v):
            return (comp_adj[v] & P).bit_count()
        
        vertices.sort(key=deg_in_P, reverse=True)
        
        color_map = {}
        next_color = 1
        # We need to track which colors are used by neighbors.
        # For each vertex, check its neighbors in `vertices` that have already been assigned a color.
        
        for v in vertices:
            used_colors = set()
            # Check neighbors in comp_adj that are in P and already colored
            neighbors = comp_adj[v] & P
            t = neighbors
            while t:
                lsb = t & (-t)
                u = lsb.bit_length() - 1
                if u in color_map:
                    used_colors.add(color_map[u])
                t &= t - 1
                
            # Find smallest available color
            c = 1
            while c in used_colors:
                c += 1
            color_map[v] = c
            next_color = max(next_color, c)
            
        # Create list of (vertex, color)
        result = [(v, color_map[v]) for v in vertices]
        # Sort by color
        result.sort(key=lambda x: x[1])
        return result

    def expand(R, P):
        nonlocal best_clique_size, best_clique_mask
        
        if time.time() > deadline:
            return
            
        if P == 0:
            r_size = R.bit_count()
            if r_size > best_clique_size:
                best_clique_size = r_size
                best_clique_mask = R
            return
            
        # Get color bound
        colored = greedy_color(P)
        
        # Process in reverse order (highest color first)
        # The bound for the i-th vertex (0-indexed from the end) is 
        # the number of colors used by the vertices up to that point?
        # Actually, if we have vertices sorted by color: v1(c1), v2(c2), ..., vk(ck).
        # If we pick vk, the remaining candidates are P' = P ∩ N(vk).
        # The vertices in P' are a subset of {v1...v_{k-1}}.
        # The maximum clique size in P' is at most the number of distinct colors in {v1...v_{k-1}}?
        # No, it's at most the maximum color index in that subset.
        
        # Standard MCS:
        # Iterate i from len(colored)-1 down to 0.
        # Let u = colored[i][0].
        # Let bound = colored[i][1] (the color of u).
        # If |R| + bound <= best_clique_size: break.
        # Why? Because any clique in the subgraph induced by {colored[0]...colored[i]} 
        # has size at most the number of colors in that set, which is <= colored[i][1]?
        # Yes, because the colors are 1..k, and if the max color is k, the clique size is <= k.
        
        for i in range(len(colored) - 1, -1, -1):
            u, c = colored[i]
            
            if R.bit_count() + c <= best_clique_size:
                break
                
            u_bit = 1 << u
            # New P is intersection of current P and neighbors of u in comp_adj
            # But we must also exclude vertices that are "before" u in the sorted list?
            # In MCS, we maintain P and X. 
            # When we pick u, we move u from P to X.
            # The new P for the recursive call is P ∩ N(u).
            # However, to avoid symmetry and ensure we don't pick vertices from X, 
            # we usually restrict P to vertices that are "after" u in the ordering?
            # No, the standard Tomita algorithm uses:
            # New P = P ∩ N(u)
            # New X = X ∩ N(u)  (not used in simple version)
            # And we remove u from P.
            
            # Wait, if we just do P ∩ N(u), we might pick vertices that were "before" u in the color order.
            # Is that a problem?
            # If we pick a vertex w that was before u (lower color), it's still a valid candidate.
            # The bound `c` for u was based on the set of vertices up to u.
            # If we recurse with P' = P ∩ N(u), the new coloring will be tighter.
            
            # One optimization: Only consider vertices in P that are "compatible" with the ordering?
            # Actually, the standard algorithm just uses P ∩ N(u).
            
            new_P = P & comp_adj[u]
            
            # Optimization: Remove u from new_P (it's not a neighbor of itself)
            # new_P &= ~u_bit (redundant if comp_adj doesn't have self loops, which it doesn't)
            
            expand(R | u_bit, new_P)
            
            # Move u to X (effectively remove from P for subsequent iterations in this loop)
            P &= ~u_bit
            # Note: We modified P locally. The `colored` list is static for this loop, 
            # but we are iterating over it. We just need to ensure we don't process u again.
            # Since we are iterating by index, and we removed u from P, 
            # the next iteration will pick the next vertex from the `colored` list.
            # We must ensure that if the next vertex in `colored` is not in P anymore, we skip it?
            # The `colored` list contains all vertices in the *original* P.
            # If a vertex was removed from P in a previous step of this loop, 
            # it might still be in `colored`.
            # We should check if the vertex is still in P before processing?
            # Or better, update the loop logic.
            
            # Actually, the standard implementation iterates through the sorted list.
            # If a vertex was excluded, we just skip it?
            # No, the bound `c` is still valid for the remaining set?
            # If we removed some vertices, the bound might be too loose, but still valid.
            
            # Let's add a check: if u is not in P, continue (should not happen if we manage P correctly)
            # But we DO remove u from P. So for the next iteration, the vertex `u_next` from `colored` 
            # is still in P (unless it was a neighbor of a previously selected vertex? No, we only removed `u`).
            # Wait, in the loop, we only remove `u`. We do not remove neighbors of `u` from P.
            # So all other vertices in `colored` are still in P.
            
            # So the logic holds.

    # Initial call
    # R = 0
    # P = all vertices 1..n
    all_vertices_mask = (1 << (n + 1)) - 2
    expand(0, all_vertices_mask)
    
    # Convert best_clique_mask to list
    result = []
    for i in range(1, n + 1):
        if best_clique_mask & (1 << i):
            result.append(i)
            
    return result if result else [1]

# Test locally?
# print(construct())
