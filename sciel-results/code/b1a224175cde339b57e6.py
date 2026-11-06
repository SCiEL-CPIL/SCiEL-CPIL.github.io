import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    def get_fallback():
        # Greenwood-Gleason colouring of K16 by cubic-residue classes of GF(16)
        # doubled: two copies with colour 3 on all edges between them
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        log, x = {}, 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
        return {'n': 32, 'colors': colors}

    def count_mono_triangles(adj, n):
        """Count total monochromatic triangles using bitmasks."""
        total = 0
        for k in range(4):
            for i in range(n):
                m = adj[k][i]
                while m:
                    lb = m & (-m)
                    j = lb.bit_length() - 1
                    if j > i:
                        common = adj[k][i] & adj[k][j]
                        total += common.bit_count()
                    m ^= lb
        return total // 3

    def run_local_search(adj, color, n, time_limit, rng, T_init=1.0, T_min=0.01, cooling_rate=0.9999):
        """Local search with simulated annealing to minimize monochromatic triangles."""
        current_score = count_mono_triangles(adj, n)
        if current_score == 0:
            return adj, color, 0
        
        best_score = current_score
        best_adj = [row[:] for row in adj]
        best_color = [row[:] for row in color]
        
        T = T_init
        
        while time.time() < time_limit:
            # Pick random edge
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            
            old_k = color[i][j]
            new_k = rng.randint(0, 3)
            if new_k == old_k:
                continue
            
            # Calculate delta in mono triangles
            common_old = adj[old_k][i] & adj[old_k][j]
            delta_remove = common_old.bit_count()
            
            common_new = adj[new_k][i] & adj[new_k][j]
            delta_add = common_new.bit_count()
            
            delta = delta_add - delta_remove
            
            # Accept or reject
            if delta <= 0 or rng.random() < math.exp(-delta / T):
                adj[old_k][i] ^= (1 << j)
                adj[old_k][j] ^= (1 << i)
                adj[new_k][i] |= (1 << j)
                adj[new_k][j] |= (1 << i)
                color[i][j] = new_k
                color[j][i] = new_k
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_adj = [row[:] for row in adj]
                    best_color = [row[:] for row in color]
                    if best_score == 0:
                        return best_adj, best_color, 0
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
        return best_adj, best_color, best_score

    def find_valid_coloring(n, time_limit, rng):
        """Try to find a valid coloring for K_n from scratch using SA."""
        if time.time() > time_limit:
            return None
        
        # Multiple restarts
        for restart in range(5):
            if time.time() > time_limit:
                break
            
            # Random init
            adj = [[0] * n for _ in range(4)]
            color = [[-1] * n for _ in range(n)]
            for i in range(n):
                for j in range(i + 1, n):
                    k = rng.randint(0, 3)
                    color[i][j] = k
                    color[j][i] = k
                    adj[k][i] |= (1 << j)
                    adj[k][j] |= (1 << i)
            
            # Run SA
            t_rem = time_limit - time.time()
            if t_rem < 1:
                break
            adj, color, score = run_local_search(adj, color, n, time.time() + t_rem * 0.8, rng)
            
            if score == 0:
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        colors.append(color[i][j])
                return {'n': n, 'colors': colors}
        
        return None

    def extend_coloring(base_n, base_colors, target_n, time_limit, rng):
        """
        Extend a valid coloring of K_base_n to K_target_n.
        base_colors: list of colors for K_base_n in lex order.
        """
        if base_n >= target_n:
            return base_colors[:target_n*(target_n-1)//2]
        
        # Build adjacency for base_n
        adj = [[0] * target_n for _ in range(4)]
        color = [[-1] * target_n for _ in range(target_n)]
        
        idx = 0
        for i in range(base_n):
            for j in range(i + 1, base_n):
                k = base_colors[idx]
                color[i][j] = k
                color[j][i] = k
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
                idx += 1
        
        # Extend vertex by vertex
        for v in range(base_n, target_n):
            if time.time() > time_limit - 5:
                break
                
            # For each new vertex v, determine colors to 0..v-1
            # We want to minimize triangles.
            # Greedy: for each u < v, pick color c that minimizes triangles involving (u,v).
            # A triangle (u,v,w) is mono if color(u,v)==color(v,w)==color(u,w).
            # When setting color(v,u), we look at existing neighbors of u.
            
            # Simple Greedy:
            for u in range(v):
                best_k = -1
                best_delta = float('inf')
                for k in range(4):
                    # If we color (u,v) with k, how many triangles (u,v,w) become mono?
                    # w must be < v.
                    # Triangle is mono if color(u,w)==k and color(v,w)==k.
                    # But color(v,w) is not set yet for w < v? No, we are setting v's colors to all u < v.
                    # We should set all colors for v at once to be consistent?
                    # Or sequentially? Sequentially is simpler but might be suboptimal.
                    # Let's try to assign all edges (v, u) for u < v simultaneously using a small local search or greedy.
                    pass
                
            # Better: Assign all edges (v, u) for u in [0, v-1] using a mini-SA or greedy.
            # Let's use a greedy approach with backtracking if stuck, or just greedy + local repair.
            
            # Initialize colors for (v, u) randomly or greedily
            colors_for_v = [0] * v
            for u in range(v):
                # Pick color that minimizes immediate conflict with existing graph
                # Conflict for (v,u) color k: number of w < v such that color(u,w)==k AND color(v,w) is eventually k?
                # This is hard to predict.
                # Let's just pick randomly first, then repair.
                colors_for_v[u] = rng.randint(0, 3)
            
            # Apply random colors
            for u in range(v):
                k = colors_for_v[u]
                color[u][v] = k
                color[v][u] = k
                adj[k][u] |= (1 << v)
                adj[k][v] |= (1 << u)
            
            # Now, run a local search restricted to edges incident to v?
            # Or just run global SA for a short time to fix triangles.
            # Global SA is safer.
            t_rem = time_limit - time.time() - 5
            if t_rem > 1:
                adj, color, score = run_local_search(adj, color, v + 1, time.time() + t_rem * 0.5, rng)
                if score == 0:
                    # Success, continue to next vertex
                    continue
                else:
                    # If not 0, we might be stuck. 
                    # If we can't fix it, we might need to restart the extension from base_n?
                    # Or accept the error and try to fix later?
                    # For now, if we get stuck, we break and return the best valid n we had.
                    # But we want to maximize n.
                    # Let's try a few restarts of the last vertex assignment.
                    success = False
                    for _ in range(3):
                        # Re-randomize v's colors
                        for u in range(v):
                            old_k = color[u][v]
                            adj[old_k][u] ^= (1 << v)
                            adj[old_k][v] ^= (1 << u)
                        
                        for u in range(v):
                            k = rng.randint(0, 3)
                            color[u][v] = k
                            color[v][u] = k
                            adj[k][u] |= (1 << v)
                            adj[k][v] |= (1 << u)
                        
                        t_rem2 = time_limit - time.time() - 5
                        if t_rem2 < 1:
                            break
                        adj, color, score = run_local_search(adj, color, v + 1, time.time() + t_rem2 * 0.4, rng)
                        if score == 0:
                            success = True
                            break
                    
                    if not success:
                        # Failed to extend to v+1. Return coloring for v vertices.
                        # Extract colors for 0..v-1
                        final_n = v
                        final_colors = []
                        for i in range(final_n):
                            for j in range(i + 1, final_n):
                                final_colors.append(color[i][j])
                        return final_colors
            else:
                # No time left
                break
        
        # If we made it to target_n
        final_n = target_n
        final_colors = []
        for i in range(final_n):
            for j in range(i + 1, final_n):
                final_colors.append(color[i][j])
        return final_colors

    # 1. Try to find a valid n=50 coloring quickly.
    # If we find it, we use it as base.
    # If not, we use n=32 base.
    
    base_n = 32
    base_colors = None
    
    # Try to find n=50 in 10 seconds
    rng = random.Random(42)
    res50 = find_valid_coloring(50, start_time + 10.0, rng)
    if res50:
        base_n = 50
        base_colors = res50['colors']
    
    if base_colors is None:
        fb = get_fallback()
        base_n = fb['n']
        base_colors = fb['colors']
    
    # 2. Try to extend to n=62
    target_n = 62
    ext_colors = extend_coloring(base_n, base_colors, target_n, deadline, rng)
    
    if ext_colors is not None:
        # Determine the actual n achieved
        # extend_coloring returns a list of colors.
        # We need to figure out n from the length?
        # len = n*(n-1)/2.
        L = len(ext_colors)
        # Solve n^2 - n - 2L = 0
        # n = (1 + sqrt(1 + 8L)) / 2
        n = int((1 + math.sqrt(1 + 8 * L)) / 2)
        
        # Verify validity
        adj = [[0] * n for _ in range(4)]
        for i in range(n):
            for j in range(i + 1, n):
                idx = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
                k = ext_colors[idx]
                adj[k][i] |= (1 << j)
                adj[k][j] |= (1 << i)
        
        if count_mono_triangles(adj, n) == 0:
            return {'n': n, 'colors': ext_colors}
    
    # 3. Fallback to base_n if extension failed or was invalid
    return {'n': base_n, 'colors': base_colors}
