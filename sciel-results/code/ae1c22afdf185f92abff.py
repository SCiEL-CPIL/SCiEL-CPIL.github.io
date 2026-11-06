import time
import random

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Build GF(16) log table for cubic residue coloring
    def _mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    
    log = {}
    x = 1
    for e in range(15):
        log[x] = e
        x = _mul(x, 2)
    
    def build_k32_colors():
        """Greenwood-Gleason K32: two copies of K16, color 3 between them."""
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return colors
    
    def build_color_bitmasks(colors, n):
        """
        For each color c, build a list of bitmasks:
        bitmask[c][i] = bitmask of vertices j such that edge (min(i,j), max(i,j)) has color c
        """
        bitmasks = [[0] * n for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                bitmasks[c][i] |= (1 << j)
                bitmasks[c][j] |= (1 << i)
                idx += 1
        return bitmasks
    
    def is_independent_set_in_color(mask, vertex_set, color, n):
        """
        Check if vertex_set (bitmask) is an independent set in the graph of edges with given color.
        For each vertex v in vertex_set, check that no other vertex in vertex_set is connected to v by this color.
        """
        # For each vertex in the set, its color-neighborhood intersected with the set should be empty
        # We need bitmasks for this color
        return True  # placeholder, will be replaced
    
    def check_extendable(bitmasks, n, new_vertex_colors):
        """
        Check if adding a new vertex with given colors to existing vertices is valid.
        new_vertex_colors[i] = color of edge (new_vertex, i)
        For each color c, the set of vertices i with new_vertex_colors[i] = c must be
        an independent set in the c-colored subgraph of the existing K_n.
        """
        for c in range(4):
            # Get set of vertices connected to new vertex with color c
            verts = 0
            for i in range(n):
                if new_vertex_colors[i] == c:
                    verts |= (1 << i)
            # Check independence: for each vertex v in verts, no other vertex in verts
            # should be connected to v by color c
            # Use bitmasks[c][v] & (verts ^ (1<<v)) should be 0 for all v in verts
            v = verts
            while v:
                low_bit = v & (-v)
                idx = low_bit.bit_length() - 1
                v ^= low_bit
                # Check if any other vertex in verts is connected to idx by color c
                if bitmasks[c][idx] & (verts ^ low_bit):
                    return False
        return True
    
    def try_add_vertex_fast(bitmasks, n, new_vertex_colors):
        """Fast check using precomputed bitmasks."""
        for c in range(4):
            verts = 0
            for i in range(n):
                if new_vertex_colors[i] == c:
                    verts |= (1 << i)
            if verts == 0:
                continue
            # Check independence
            v = verts
            while v:
                low_bit = v & (-v)
                idx = low_bit.bit_length() - 1
                v ^= low_bit
                if bitmasks[c][idx] & (verts ^ low_bit):
                    return False
        return True
    
    def extend_coloring(base_colors, base_n, target_n, time_limit, rng):
        """
        Try to extend base coloring from base_n to target_n vertices.
        Returns colors list if successful, else None.
        """
        colors = base_colors[:]
        current_n = base_n
        bitmasks = build_color_bitmasks(colors, current_n)
        
        start_time = time.time()
        
        for new_v in range(base_n, target_n):
            vertex_found = False
            # Try random assignments
            while time.time() < start_time + time_limit:
                if not vertex_found:
                    # Generate random color assignment for new vertex
                    new_colors = [rng.randint(0, 3) for _ in range(current_n)]
                    if try_add_vertex_fast(bitmasks, current_n, new_colors):
                        # Valid! Add the vertex
                        colors.extend(new_colors)
                        current_n += 1
                        # Rebuild bitmasks
                        bitmasks = build_color_bitmasks(colors, current_n)
                        vertex_found = True
                        break
                else:
                    break
            
            if not vertex_found:
                # Try more systematically: fix some colors, randomize others
                # Use the structure: vertices 0-15 are copy A, 16-31 are copy B
                # New vertex should connect to A and B with specific patterns
                success = False
                for attempt in range(10000):
                    if time.time() > start_time + time_limit:
                        break
                    # Strategy: color to A from {0,1,2}, color to B from {0,1,2} 
                    # (avoid color 3 to both A and B to prevent mono triangles via cross edges)
                    new_colors = []
                    for i in range(16):
                        new_colors.append(rng.randint(0, 2))
                    for i in range(16, 32):
                        new_colors.append(rng.randint(0, 2))
                    # For vertices 32+, random
                    for i in range(32, current_n):
                        new_colors.append(rng.randint(0, 3))
                    
                    if try_add_vertex_fast(bitmasks, current_n, new_colors):
                        colors.extend(new_colors)
                        current_n += 1
                        bitmasks = build_color_bitmasks(colors, current_n)
                        success = True
                        break
                
                if not success:
                    return None
        
        return colors
    
    # Start with K32
    base_colors = build_k32_colors()
    base_n = 32
    rng = random.Random(42)
    
    # Try to extend to 50
    best_n = 32
    best_colors = base_colors[:]
    
    # Try extensions with different seeds and time allocations
    for target in [50, 49, 48, 47, 46, 45]:
        if time.time() > deadline - 10:
            break
        
        # Multiple attempts with different seeds
        for seed in range(5):
            if time.time() > deadline - 5:
                break
            rng_attempt = random.Random(seed * 1000 + 42)
            time_left = min(20, deadline - time.time() - 2)
            if time_left < 1:
                break
            
            result = extend_coloring(base_colors, base_n, target, time_left, rng_attempt)
            if result is not None:
                return {'n': target, 'colors': result}
    
    # If extension failed, try a different approach: direct SA on smaller n
    # But first, try to extend to just 33, 34, etc.
    for target in range(33, 40):
        if time.time() > deadline - 5:
            break
        for seed in range(3):
            if time.time() > deadline - 3:
                break
            rng_attempt = random.Random(seed * 7 + 13)
            time_left = min(5, deadline - time.time() - 2)
            if time_left < 0.5:
                break
            result = extend_coloring(base_colors, base_n, target, time_left, rng_attempt)
            if result is not None:
                return {'n': target, 'colors': result}
    
    # Fallback: return K32
    return {'n': 32, 'colors': best_colors}
