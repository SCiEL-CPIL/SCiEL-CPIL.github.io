import time

def construct():
    start_time = time.time()
    # Safety margin: The limit is 90s. Let's aim to finish by 85s to be safe.
    timeout = 85.0
    
    n = 256
    vertices = range(1, n + 1)
    
    # 1. Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    # We want the Maximum Independent Set in G, which is the Maximum Clique in G_complement.
    
    # adjacency bitmask for G (conflicts)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    # adj_comp[v] contains bits for all u != v such that (a*b) % (a+b) != 0
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
    best_size = 0
    best_set_mask = 0
    
    # Helper to get list of vertices from a bitmask
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
    # We color the subgraph induced by candidates in G_comp.
    # Returns a list of (vertex, color) sorted by color descending.
    def greedy_color(candidates_mask):
        verts = get_vertices(candidates_mask)
        
        # Calculate degrees within candidates
        degrees = []
        for v in verts:
            deg = (adj_comp[v] & candidates_mask).bit_count()
            degrees.append((deg, v))
            
        # Sort by degree descending
        degrees.sort(reverse=True)
        ordered_verts = [v for _, v in degrees]
        
        # Use bitmask-based color tracking for speed
        colors = {}
        color_masks = [0]  # color_masks[c] = bitmask of vertices assigned color c
        max_color = 0
        
        for v in ordered_verts:
            # Find smallest color c such that no neighbor of v has color c
            c = 1
            while True:
                if c < len(color_masks):
                    if (adj_comp[v] & color_masks[c]) == 0:
                        break
                else:
                    color_masks.append(0)
                    break
                c += 1
            
            colors[v] = c
            color_masks[c] |= (1 << v)
            if c > max_color:
                max_color = c
        
        # Create list of (vertex, color) sorted by color descending
        color_list = [(v, colors[v]) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    def expand(candidates_mask, current_size, current_mask):
        nonlocal best_size, best_set_mask, start_time
        
        # Check timeout
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
            
        # P is the set of candidates we will iterate over
        # We process vertices in the order given by color_list (descending color)
        # However, we must maintain a working set P that shrinks as we iterate
        # to ensure we don't pick the same vertex in different branches at this level?
        # Actually, the standard Tomita loop is:
        # P = candidates
        # for v in ordered_vertices:
        #    if v not in P: continue (shouldn't happen if we manage P correctly)
        #    if current_size + color(v) <= best_size: break
        #    recurse(P & N(v))
        #    P.remove(v)
        
        P = candidates_mask
        
        for v, c in color_list:
            # If the vertex v is no longer in P (removed by previous iterations? No, 
            # we only remove the vertex we just processed. But wait, 
            # the color_list is static for this call. 
            # If we remove v from P, subsequent vertices in the list are still in P 
            # unless they were removed? No, we only remove the specific v we just branched on.
            # So v should always be in P at the start of its iteration?
            # Yes, because we haven't removed it yet.
            
            # Check if v is still in P (it should be, unless there's a bug in logic)
            if not (P & (1 << v)):
                continue
                
            # Prune if bound is too low
            if current_size + c <= best_size:
                break
                
            # Recurse: Pick v
            # New candidates are neighbors of v in P
            new_candidates = P & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            # Don't pick v: Remove v from P
            P &= ~(1 << v)

    # Initial heuristic: try multiple greedy strategies for better lower bound
    def greedy_clique(start_mask):
        candidates = start_mask
        clique_mask = 0
        size = 0
        while candidates:
            verts = get_vertices(candidates)
            best_v = -1
            best_deg = -1
            for v in verts:
                deg = (adj_comp[v] & candidates).bit_count()
                if deg > best_deg:
                    best_deg = deg
                    best_v = v
            if best_v == -1:
                break
            clique_mask |= (1 << best_v)
            size += 1
            candidates = candidates & adj_comp[best_v]
        return size, clique_mask
    
    # Strategy 1: greedy from full set
    s1, m1 = greedy_clique(all_vertices_mask)
    
    # Strategy 2: try removing each vertex and greedy from remaining (take best)
    best_s = s1
    best_m = m1
    for v in vertices:
        if time.time() - start_time > timeout - 5:
            break
        mask_without_v = all_vertices_mask & ~(1 << v)
        s, m = greedy_clique(mask_without_v)
        if s > best_s:
            best_s = s
            best_m = m
    
    best_size = best_s
    best_set_mask = best_m

    # Start the search
    expand(all_vertices_mask, 0, 0)
    
    # Extract result
    result_set = set()
    mask = best_set_mask
    while mask:
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        result_set.add(v)
        mask ^= lsb
        
    return list(result_set)

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
