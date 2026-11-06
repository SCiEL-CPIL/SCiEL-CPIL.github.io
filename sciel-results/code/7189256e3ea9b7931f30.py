import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin from 90s limit
    
    n = 256
    vertices = range(1, n + 1)
    
    # 1. Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    # We will solve for Maximum Independent Set in G.
    # This is equivalent to Maximum Clique in G_complement.
    
    # adjacency bitmask for G (conflicts)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    # A vertex u is a neighbor of v in G_comp if u != v and u is NOT a neighbor of v in G.
    # Note: In G, v is not connected to itself. In G_comp, v is not connected to itself.
    # So for v, neighbors in G_comp are all u in [1, n] such that u != v and ((1 << u) & adj_conflict[v]) == 0.
    
    adj_comp = [0] * (n + 1)
    all_vertices_mask = 0
    for i in range(1, n + 1):
        all_vertices_mask |= (1 << i)
        
    for v in vertices:
        # Start with all vertices, remove v itself, remove conflict neighbors
        mask = all_vertices_mask & ~(1 << v)
        mask &= ~adj_conflict[v]
        adj_comp[v] = mask
        
    # Tomita Algorithm for Maximum Clique
    # We use bitmasks.
    # best_size: size of the best clique found so far
    best_size = 0
    best_set_mask = 0
    
    def get_vertices(mask):
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
        return verts

    # Greedy coloring for upper bound
    # Returns list of (vertex, color_index) sorted by color_index descending?
    # Actually, Tomita uses a specific ordering.
    # We color the subgraph induced by `candidates` in G_comp.
    # The color of a vertex is 1 + max(color of its colored neighbors).
    # The maximum color used is an upper bound on the clique size in that subgraph.
    # We also assign colors to each vertex to allow pruning.
    
    def greedy_color(candidates_mask):
        # Returns a list of (vertex, color) for all vertices in candidates_mask
        # Vertices are ordered by degree descending to help the bound.
        
        colors = {}
        
        # To make the bound tighter, we order vertices by degree in the candidate subgraph descending.
        verts = get_vertices(candidates_mask)
        
        # Degree calculation for each v in verts within candidates_mask
        degrees = []
        for v in verts:
            # Count neighbors in candidates_mask
            deg = (adj_comp[v] & candidates_mask).bit_count()
            degrees.append((deg, v))
            
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        max_color = 0
        colored_mask = 0
        
        for v in ordered_verts:
            # Find the smallest color c such that v has no neighbor with color c in colored_mask
            used_colors = set()
            # Iterate bits of neighbors in colored set directly to avoid list creation
            m = adj_comp[v] & colored_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                used_colors.add(colors[u])
                m ^= lsb
                    
            c = 1
            while c in used_colors:
                c += 1
                
            colors[v] = c
            if c > max_color:
                max_color = c
            colored_mask |= (1 << v)
            
        # Return list of (vertex, color)
        color_list = [(v, colors[v]) for v in ordered_verts]
        # Sort by color descending to prune earlier
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Get upper bound via coloring
        color_list, max_color = greedy_color(candidates_mask)
        
        # If the upper bound doesn't exceed best, prune
        if current_size + max_color <= best_size:
            return
            
        # Iterate through vertices in color_list (which is sorted by color desc)
        # We need to maintain the candidate set as we go?
        # The standard recursive step:
        # Pick a vertex v from the list.
        # New candidates = candidates intersect neighbors(v)
        
        # However, the coloring is done on the *initial* candidate set.
        # When we recurse, the candidate set shrinks.
        # We must be careful: the color bound for vertex v is valid for the subgraph
        # consisting of v and its neighbors in the *current* candidate set?
        # Actually, the bound `color(v)` is an upper bound on the clique size in the subgraph
        # induced by {v} U {u in candidates | u is adjacent to v}.
        # Wait, the coloring was on the whole `candidates_mask`.
        # The property is: if we color the graph G[C], then any clique in G[C] can have at most
        # max_color vertices.
        # More specifically, if we order vertices v1, ..., vk by color c1, ..., ck,
        # and we pick v_i, the max clique size in the branch v_i is bounded by c_i + current_size?
        # Yes, if the colors are assigned such that neighbors have different colors,
        # and we process in decreasing color order, then the vertices available in the branch
        # for v_i are a subset of the vertices with color <= c_i that are neighbors of v_i.
        # Actually, simpler: The maximum clique size in the subgraph induced by
        # {v_i} U {v_j | j > i, v_j adjacent to v_i} is at most c_i.
        # So if current_size + c_i <= best_size, we can prune.
        
        for v, c in color_list:
            if current_size + c <= best_size:
                break
                
            # Check if v is still in candidates?
            # The color_list is generated from the original candidates_mask.
            # As we iterate, we don't remove v from candidates_mask for the *current* iteration's bound check,
            # but for the *next* iterations?
            # In Tomita, we usually modify the candidate set.
            # Let's stick to the standard recursive formulation:
            # P = candidates
            # while P is not empty:
            #   choose v in P
            #   P = P \ {v}
            #   ...
            
            # My `color_list` is a static list derived from the start of `expand`.
            # This is slightly different from the dynamic P reduction in some implementations.
            # However, if we just check `current_size + c <= best_size`, it's a valid bound
            # because `c` is the color of `v` in the coloring of the *original* candidate set.
            # Is it safe?
            # If we removed vertices from P, the coloring might be too loose, but not invalid.
            # The clique in the new subgraph is a clique in the old subgraph.
            # So the bound holds.
            
            # Recurse
            # New candidates: neighbors of v in the current candidate set
            # But we must ensure we don't pick vertices that were "before" v in some order?
            # No, for Maximum Clique, we just need the intersection with neighbors.
            # To avoid duplicates and ensure we explore each clique once,
            # we typically define an order and only pick vertices later in the order?
            # Or simply:
            # new_candidates = candidates_mask & adj_comp[v]
            # But this might include vertices that are not "available" in a strict partition sense?
            # No, for Max Clique, if we pick v, we MUST pick all other vertices from N(v).
            # So new_candidates = candidates_mask & adj_comp[v] is correct.
            
            # However, to prune effectively and avoid re-checking, we often remove v from the pool
            # for subsequent iterations in the *same* level?
            # The loop `for v, c in color_list` iterates over a fixed list.
            # If we pick v, we recurse. When we come back, we pick the next v'.
            # The candidates for v' should be the *original* candidates minus {v}?
            # No, the standard Tomita loop is:
            # P = candidates
            # while P:
            #   v = select(P)
            #   P.remove(v)
            #   expand(P & N(v))
            
            # My current structure iterates over a pre-computed list.
            # This is equivalent to:
            # P = candidates
            # for v in ordered_P:
            #    if v in P:
            #       P.remove(v)
            #       expand(P & N(v))
            
            # I need to maintain a `remaining` mask or similar.
            
            pass # Placeholder for logic correction below

    # Corrected Expand Logic
    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        if time.time() - start_time > timeout:
            return
            
        if candidates_mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_set_mask = current_mask
            return
            
        # Get upper bound
        color_list, max_color = greedy_color(candidates_mask)
        
        if current_size + max_color <= best_size:
            return
            
        # We need to iterate and remove vertices from a working set of candidates
        # to ensure we don't pick the same vertex twice in different branches at this level?
        # Actually, the branches are disjoint if we partition P correctly.
        # Standard:
        # P = candidates_mask
        # We process vertices in order of decreasing color.
        
        # Let's reconstruct the logic to match Tomita exactly for safety
        # 1. Color P to get colors and order.
        # 2. Iterate v in order.
        # 3. If bound is too low, stop.
        # 4. Recurse with P & N(v).
        # 5. Remove v from P.
        
        # Since greedy_color returns a list of all vertices in candidates_mask,
        # we can use that.
        
        P = candidates_mask
        # The color_list contains (v, color).
        # We iterate through this list.
        # But we must only process v if it is still in P?
        # Actually, if we remove v from P, subsequent vertices in the list that were in P
        # are still valid candidates for *other* branches?
        # No, in the Max Clique algorithm, once we decide to NOT include v in the current path,
        # we remove it from P so it's not considered for other vertices in this level.
        # Wait, if we pick v, we go to N(v). If we DON'T pick v, we proceed to next vertex.
        # The "Don't pick v" case is handled by removing v from P and continuing the loop.
        # The "Pick v" case is handled by the recursive call.
        
        for v, c in color_list:
            if current_size + c <= best_size:
                break
                
            # Check if v is still in P
            if not (P & (1 << v)):
                continue
                
            # Recurse: Pick v
            # New candidates are neighbors of v in P
            new_candidates = P & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            # Don't pick v: Remove v from P
            P &= ~(1 << v)

    # Start the search
    # Initial candidates: all vertices
    initial_candidates = all_vertices_mask
    expand(initial_candidates, 0, 0)
    
    # Extract result
    result_set = set()
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        result_set.add(v)
        mask ^= lsb
        
    return list(result_set)

# Run once to verify it works and time it
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
