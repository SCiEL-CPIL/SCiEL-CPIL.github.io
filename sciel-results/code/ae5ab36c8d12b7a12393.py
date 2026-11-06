import time

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin from 90s limit
    
    n = 256
    vertices = range(1, n + 1)
    
    # Build the conflict graph G (edges where (a*b) % (a+b) == 0)
    adj_conflict = [0] * (n + 1)
    
    for a in vertices:
        for b in vertices:
            if a < b:
                if (a * b) % (a + b) == 0:
                    adj_conflict[a] |= (1 << b)
                    adj_conflict[b] |= (1 << a)
                    
    # Build complement graph adjacency bitmasks
    adj_comp = [0] * (n + 1)
    all_vertices_mask = 0
    for i in range(1, n + 1):
        all_vertices_mask |= (1 << i)
        
    for v in vertices:
        mask = all_vertices_mask & ~(1 << v)
        mask &= ~adj_conflict[v]
        adj_comp[v] = mask
        
    # Greedy initial solution: pick vertices with fewest conflicts first
    best_size = 0
    best_set_mask = 0
    
    # Greedy construction for initial lower bound
    greedy_set = set()
    remaining = set(vertices)
    for v in sorted(vertices, key=lambda x: (adj_conflict[x] & all_vertices_mask).bit_count()):
        if all((v * u) % (v + u) != 0 for u in greedy_set):
            greedy_set.add(v)
    
    if greedy_set:
        best_size = len(greedy_set)
        for v in greedy_set:
            best_set_mask |= (1 << v)
    
    # Fast bit-parallel greedy coloring
    # Colors vertices in candidates_mask, returns list of (vertex, color) sorted by color desc
    # Uses array-based color assignment for speed
    
    def greedy_color(candidates_mask):
        """
        Returns (color_list, max_color) where color_list is [(v, c), ...] sorted by c desc.
        Uses sequential coloring with degree-based ordering.
        """
        if candidates_mask == 0:
            return [], 0
        
        # Get vertices and order by degree in candidate subgraph
        verts = []
        m = candidates_mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            deg = (adj_comp[v] & candidates_mask).bit_count()
            verts.append((deg, v))
            m ^= lsb
        
        verts.sort(reverse=True)
        ordered_verts = [v for _, v in verts]
        
        # Color assignment using arrays
        color_of = [0] * (n + 1)
        max_color = 0
        
        for v in ordered_verts:
            # Find used colors among colored neighbors
            used = [False] * 64  # Max 256 vertices, so max 256 colors, but typically much less
            m = adj_comp[v] & candidates_mask
            while m:
                lsb = m & -m
                u = lsb.bit_length() - 1
                if color_of[u] > 0:
                    c = color_of[u]
                    if c < len(used):
                        used[c] = True
                    else:
                        used.extend([False] * (c + 1 - len(used)))
                        used[c] = True
                m ^= lsb
            
            c = 1
            while c < len(used) and used[c]:
                c += 1
            if c >= len(used):
                used.append(True)
            color_of[v] = c
            if c > max_color:
                max_color = c
        
        # Build result list sorted by color descending
        color_list = [(v, color_of[v]) for v in ordered_verts]
        color_list.sort(key=lambda x: x[1], reverse=True)
        
        return color_list, max_color

    # Branch and bound for maximum clique
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
        
        if current_size + max_color <= best_size:
            return
            
        P = candidates_mask
        
        for v, c in color_list:
            if current_size + c <= best_size:
                break
                
            if not (P & (1 << v)):
                continue
                
            # Recurse: include v
            new_candidates = P & adj_comp[v]
            expand(new_candidates, current_size + 1, current_mask | (1 << v))
            
            # Exclude v
            P &= ~(1 << v)

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
