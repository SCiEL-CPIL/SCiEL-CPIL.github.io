import time
import random

def construct():
    deadline = time.time() + 115

    # --- Base n=32 colouring (Greenwood-Gleason) ---
    def mul(a, b):
        r = 0
        for i in range(4):
            if (b >> i) & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if (r >> i) & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log_map = {}
    x = 1
    for e in range(15):
        log_map[x] = e
        x = mul(x, 2)

    def edge_color_core(i, j):
        return log_map[i ^ j] % 3

    n_base = 32
    base_colors = [0] * (n_base * (n_base - 1) // 2)
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            if i // 16 == j // 16:
                c = edge_color_core(i % 16, j % 16)
            else:
                c = 3
            base_colors[idx] = c
            idx += 1

    # --- Local Search for n=50 ---
    N = 50
    num_edges = N * (N - 1) // 2
    
    # Edge mapping
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))
            
    # Triangle mapping: for each edge, list of triangles (indices in a list?)
    # Actually, for an edge (u, v), the triangles are (u, v, w) for w in 0..N-1, w!=u,v.
    # We can just iterate w in delta_change.
    
    # To maintain a list of mono triangles, we need to know which triangles are mono.
    # A triangle is identified by (i, j, k) with i<j<k.
    # Let's precompute for each edge idx, the list of (u, v, w) triples?
    # No, just (w, idx_uw, idx_vw) is enough to check.
    
    # Precompute neighbors for each edge
    # edge_neighbors[idx] = list of (w, idx_uw, idx_vw)
    edge_neighbors = [[] for _ in range(num_edges)]
    
    for idx, (u, v) in enumerate(edges):
        for w in range(N):
            if w == u or w == v:
                continue
            # Triangle (u, v, w)
            # Need indices for (u, w) and (v, w)
            # Ensure min/max for key
            if u < w:
                idx_uw = edge_to_idx[(u, w)]
            else:
                idx_uw = edge_to_idx[(w, u)]
                
            if v < w:
                idx_vw = edge_to_idx[(v, w)]
            else:
                idx_vw = edge_to_idx[(w, v)]
                
            edge_neighbors[idx].append((w, idx_uw, idx_vw))

    def init_mono_list(colors):
        mono_set = set()
        # Iterate all triangles
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                idx_ij = edge_to_idx[(i, j)]
                c_ij = colors[idx_ij]
                for k in range(j + 1, N):
                    idx_jk = edge_to_idx[(j, k)]
                    idx_ik = edge_to_idx[(i, k)]
                    if colors[idx_jk] == c_ij and colors[idx_ik] == c_ij:
                        # Mono triangle
                        # Store as (i, j, k)
                        mono_set.add((i, j, k))
        return list(mono_set)

    def update_mono_list(mono_list, mono_set, edge_idx, old_c, new_c):
        # Remove triangles that were mono with old_c but aren't with new_c
        # Add triangles that aren't mono with old_c but are with new_c
        # Only triangles involving edge_idx are affected.
        
        # We need to check all triangles involving this edge.
        # The edge is (u, v).
        u, v = edges[edge_idx]
        
        # Triangles are (u, v, w)
        for w, idx_uw, idx_vw in edge_neighbors[edge_idx]:
            # Triangle vertices: u, v, w
            # Sort them for consistent key (i, j, k)
            verts = sorted([u, v, w])
            tri_key = tuple(verts)
            
            # Check if it was mono
            was_mono = tri_key in mono_set
            
            # Check if it is mono now
            # Colors:
            # Edge (u, v) is now new_c
            # Edge (u, w) is colors[idx_uw]
            # Edge (v, w) is colors[idx_vw]
            
            c_uw = colors[idx_uw]
            c_vw = colors[idx_vw]
            
            is_mono_now = (new_c == c_uw == c_vw)
            
            if was_mono and not is_mono_now:
                mono_set.remove(tri_key)
                # Remove from list? List removal is O(N). 
                # Better to use a set for membership and a list for iteration?
                # Or just rebuild list when needed?
                # Since we pop from list, we can just leave dead entries and skip them?
                # Or use a set and pick random from set?
                # Set pick is O(1).
                pass
            
            elif not was_mono and is_mono_now:
                mono_set.add(tri_key)

        # We are using a set `mono_set` to track current mono triangles.
        # We don't strictly need a list if we can pick a random item from the set.
        # `random.choice(list(mono_set))` is O(S).
        # We can just keep the set.
        
    # Local Search Loop
    best_n = n_base
    best_colors = base_colors[:]
    
    attempts = 0
    while time.time() < deadline and attempts < 20:
        # Random init
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Init mono set
        mono_set = set()
        # Efficient init:
        for i in range(N - 2):
            for j in range(i + 1, N - 1):
                idx_ij = edge_to_idx[(i, j)]
                c_ij = colors[idx_ij]
                for k in range(j + 1, N):
                    idx_jk = edge_to_idx[(j, k)]
                    idx_ik = edge_to_idx[(i, k)]
                    if colors[idx_jk] == c_ij and colors[idx_ik] == c_ij:
                        mono_set.add((i, j, k))
                        
        if len(mono_set) == 0:
            return {'n': N, 'colors': colors}
            
        temperature = 1.0
        min_temp = 0.05
        cooling = 0.999
        steps = 0
        max_steps = 50000
        
        while len(mono_set) > 0 and steps < max_steps and time.time() < deadline:
            # Pick a random mono triangle
            tri = random.choice(list(mono_set))
            i, j, k = tri
            
            # Pick a random edge from the triangle
            # Edges: (i,j), (j,k), (i,k)
            edge_choice = random.randint(0, 2)
            if edge_choice == 0:
                u, v = i, j
            elif edge_choice == 1:
                u, v = j, k
            else:
                u, v = i, k
                
            idx_uv = edge_to_idx[(min(u, v), max(u, v))]
            old_c = colors[idx_uv]
            
            # Try all 4 colors
            best_delta = float('inf')
            best_new_c = -1
            
            # We need to calculate delta for each new_c
            # Delta = (new mono created) - (old mono destroyed)
            # We can just check the neighbors of this edge.
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                
                created = 0
                destroyed = 0
                
                for w, idx_uw, idx_vw in edge_neighbors[idx_uv]:
                    # Triangle (u, v, w)
                    verts = sorted([u, v, w])
                    tri_key = tuple(verts)
                    
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    
                    # Was it mono?
                    was_mono = (old_c == c_uw == c_vw)
                    # Is it mono with new_c?
                    is_mono_new = (new_c == c_uw == c_vw)
                    
                    if was_mono:
                        destroyed += 1
                    if is_mono_new:
                        created += 1
                        
                delta = created - destroyed
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
                    
            if best_new_c == -1:
                break
                
            # Acceptance criterion
            if best_delta < 0:
                # Improve
                # Update colors
                colors[idx_uv] = best_new_c
                # Update mono_set
                # We need to update mono_set for all triangles involving idx_uv
                u, v = edges[idx_uv]
                for w, idx_uw, idx_vw in edge_neighbors[idx_uv]:
                    verts = sorted([u, v, w])
                    tri_key = tuple(verts)
                    
                    c_uw = colors[idx_uw]
                    c_vw = colors[idx_vw]
                    new_c = best_new_c
                    
                    # Re-evaluate
                    is_mono = (new_c == c_uw == c_vw)
                    
                    if is_mono:
                        mono_set.add(tri_key)
                    else:
                        # It might have been in the set
                        if tri_key in mono_set:
                            mono_set.remove(tri_key)
                            
                steps += 1
            elif best_delta == 0:
                if random.random() < 0.3:
                    colors[idx_uv] = best_new_c
                    # Update mono_set similarly
                    u, v = edges[idx_uv]
                    for w, idx_uw, idx_vw in edge_neighbors[idx_uv]:
                        verts = sorted([u, v, w])
                        tri_key = tuple(verts)
                        c_uw = colors[idx_uw]
                        c_vw = colors[idx_vw]
                        new_c = best_new_c
                        is_mono = (new_c == c_uw == c_vw)
                        if is_mono:
                            mono_set.add(tri_key)
                        else:
                            if tri_key in mono_set:
                                mono_set.remove(tri_key)
                    steps += 1
            else:
                # Worsening
                if random.random() < pow(2.71828, -best_delta / temperature):
                    colors[idx_uv] = best_new_c
                    # Update mono_set
                    u, v = edges[idx_uv]
                    for w, idx_uw, idx_vw in edge_neighbors[idx_uv]:
                        verts = sorted([u, v, w])
                        tri_key = tuple(verts)
                        c_uw = colors[idx_uw]
                        c_vw = colors[idx_vw]
                        new_c = best_new_c
                        is_mono = (new_c == c_uw == c_vw)
                        if is_mono:
                            mono_set.add(tri_key)
                        else:
                            if tri_key in mono_set:
                                mono_set.remove(tri_key)
                    steps += 1
            
            temperature *= cooling
            if temperature < min_temp:
                temperature = min_temp

        attempts += 1

    return {'n': best_n, 'colors': best_colors}
