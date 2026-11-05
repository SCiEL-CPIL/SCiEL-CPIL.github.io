import time
import random

def construct():
    deadline = time.time() + 115
    N = 50
    num_edges = N * (N - 1) // 2

    # Edge mapping
    edges = []
    edge_to_idx = {}
    for i in range(N):
        for j in range(i + 1, N):
            edge_to_idx[(i, j)] = len(edges)
            edges.append((i, j))

    # Precompute: for each edge (u,v), list of (w, idx_uw, idx_vw)
    edge_neighbors = [[] for _ in range(num_edges)]
    for idx, (u, v) in enumerate(edges):
        for w in range(N):
            if w == u or w == v:
                continue
            if u < w:
                idx_uw = edge_to_idx[(u, w)]
            else:
                idx_uw = edge_to_idx[(w, u)]
            if v < w:
                idx_vw = edge_to_idx[(v, w)]
            else:
                idx_vw = edge_to_idx[(w, v)]
            edge_neighbors[idx].append((w, idx_uw, idx_vw))

    # Build neighbor bitmasks for each vertex and color
    # For vertex v, color c: bitmask of neighbors w such that edge (v,w) has color c
    def build_color_bitmasks(colors):
        """For each vertex v and color c, build a bitmask of neighbors connected by color c."""
        # neighbor_mask[v][c] = bitmask of vertices w where edge (v,w) has color c
        neighbor_mask = [[0] * 4 for _ in range(N)]
        for idx, (u, v) in enumerate(edges):
            c = colors[idx]
            neighbor_mask[u][c] |= (1 << v)
            neighbor_mask[v][c] |= (1 << u)
        return neighbor_mask

    def count_mono_triangles_bitmask(colors):
        """Count monochromatic triangles using bitmask intersection."""
        neighbor_mask = build_color_bitmasks(colors)
        count = 0
        for c in range(4):
            for i in range(N):
                mask_i = neighbor_mask[i][c]
                # For each neighbor j of i with color c, count common neighbors with color c
                m = mask_i
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    # Count common neighbors of i and j with color c, excluding i and j
                    common = mask_i & neighbor_mask[j][c]
                    # Exclude i and j themselves
                    common &= ~(1 << i) & ~(1 << j)
                    count += common.bit_count() // 2  # Each triangle counted once per vertex pair
                    m ^= lb
        return count // 3  # Each triangle counted 3 times (once per vertex)

    def count_mono_triangles_fast(colors):
        """Faster count using edge-based approach with bitmasks."""
        neighbor_mask = build_color_bitmasks(colors)
        count = 0
        for c in range(4):
            for i in range(N):
                mask_i = neighbor_mask[i][c]
                m = mask_i
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    common = mask_i & neighbor_mask[j][c]
                    common &= ~(1 << i) & ~(1 << j)
                    count += common.bit_count()
                    m ^= lb
        return count // 3

    def compute_delta_for_edge_change(colors, edge_idx, old_c, new_c):
        """Compute change in mono triangle count when changing edge (u,v) from old_c to new_c."""
        u, v = edges[edge_idx]
        # For each w, check if triangle (u,v,w) is mono in old_c or new_c
        delta = 0
        for w, idx_uw, idx_vw in edge_neighbors[edge_idx]:
            c_uw = colors[idx_uw]
            c_vw = colors[idx_vw]
            # Triangle (u,v,w) is mono in old_c if c_uw == old_c and c_vw == old_c
            if c_uw == old_c and c_vw == old_c:
                delta -= 1  # destroying a mono triangle
            # Triangle (u,v,w) is mono in new_c if c_uw == new_c and c_vw == new_c
            if c_uw == new_c and c_vw == new_c:
                delta += 1  # creating a mono triangle
        return delta

    def gg_coloring_32():
        """Generate valid 4-coloring of K_32 using Greenwood-Gleason construction."""
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
        return base_colors

    # Try simulated annealing with bitmask-based counting
    best_result = None
    best_mono = float('inf')
    
    for restart in range(200):
        if time.time() >= deadline - 5:
            break
        
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        mono_count = count_mono_triangles_fast(colors)
        
        if mono_count == 0:
            return {'n': N, 'colors': colors}
        
        temperature = 2.0
        min_temp = 0.05
        cooling = 0.9995
        steps = 0
        max_steps = 80000
        no_improve = 0
        no_improve_limit = 5000
        
        while mono_count > 0 and steps < max_steps and time.time() < deadline - 5:
            steps += 1
            edge_idx = random.randint(0, num_edges - 1)
            old_c = colors[edge_idx]
            
            best_delta = float('inf')
            best_new_c = -1
            
            for new_c in range(4):
                if new_c == old_c:
                    continue
                delta = compute_delta_for_edge_change(colors, edge_idx, old_c, new_c)
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            if best_new_c == -1:
                continue
            
            accept = False
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                if random.random() < 0.5:
                    accept = True
            else:
                if random.random() < pow(2.71828, -best_delta / temperature):
                    accept = True
            
            if accept:
                colors[edge_idx] = best_new_c
                mono_count += best_delta
                if mono_count < 0:
                    mono_count = 0
                
                if best_delta < 0:
                    no_improve = 0
                else:
                    no_improve += 1
                
                if no_improve > no_improve_limit:
                    temperature *= 0.5
                    no_improve = 0
                    if temperature < min_temp:
                        break
            
            temperature *= cooling
            if temperature < min_temp:
                temperature = min_temp
            
            if mono_count == 0:
                return {'n': N, 'colors': colors}
        
        if mono_count < best_mono:
            best_mono = mono_count
            best_result = colors[:]
    
    # If we found a valid coloring during search, return it
    if best_result is not None and best_mono == 0:
        return {'n': N, 'colors': best_result}
    
    # Fallback: return valid K_32 coloring
    fallback_colors = gg_coloring_32()
    return {'n': 32, 'colors': fallback_colors}
