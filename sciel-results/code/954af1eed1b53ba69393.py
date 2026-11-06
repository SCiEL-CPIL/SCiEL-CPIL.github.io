import random
import time

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4s using Optimized Local Search.
    """
    
    def solve_local_search(N, time_limit=100):
        start_time = time.time()
        
        # adj[c][u] is a bitmask of neighbors of u with color c
        adj = [[0] * N for _ in range(5)]
        
        # colors[u][v] is the color of edge (u,v)
        # We only need to store for u < v, but for symmetry in lookup, 
        # we can use a 2D list or just derive from adj.
        # Deriving color from adj: 
        # for c in 0..4: if (1<<v) & adj[c][u]: color is c.
        # This is O(5) per lookup. Acceptable.
        
        def assign_color(u, v, c_new):
            # Find old color
            c_old = -1
            for c in range(5):
                if (1 << v) & adj[c][u]:
                    c_old = c
                    break
            
            if c_old == c_new:
                return 0 # No change
            
            # Remove from old
            adj[c_old][u] &= ~(1 << v)
            adj[c_old][v] &= ~(1 << u)
            
            # Add to new
            adj[c_new][u] |= (1 << v)
            adj[c_new][v] |= (1 << u)
            
            return c_old

        def count_c4_through_edge(u, v, c):
            """
            Counts C4s in color c containing edge (u,v).
            Assumes (u,v) is currently NOT in adj[c] for 'potential' count,
            or IS in adj[c] for 'current' count.
            The formula is robust to both if we handle masks correctly.
            Formula: Sum over w in N_c(u) \ {v} of | N_c(w) & (N_c(v) \ {u, w}) |
            Wait, if (u,v) is in adj[c], then v is in N_c(u). We exclude it.
            u is in N_c(v). We exclude it.
            """
            mask_u = adj[c][u] & ~(1 << v)
            mask_v = adj[c][v] & ~(1 << u)
            
            count = 0
            m = mask_u
            while m:
                lsb = m & -m
                w = lsb.bit_length() - 1
                m ^= lsb
                # x must be in N_c(w) and in mask_v
                # Also x != w (impossible as no self loop)
                # Also x != u (masked out)
                # Also x != v (v is in N_c(w)? If w is neighbor of u, and v is neighbor of u.
                # If w is connected to v, then v is in N_c(w).
                # But we are counting C4s u-w-x-v.
                # If x=v, then u-w-v-v-u? No.
                # x must be distinct from u, v, w.
                # mask_v excludes u.
                # Does mask_v exclude v? No, v is not in N_c(v).
                # So x cannot be v.
                # So we just need N_c(w) & mask_v.
                common = adj[c][w] & mask_v
                count += common.bit_count()
            return count

        def delta_energy(u, v, c_new):
            # Find c_old
            c_old = -1
            for c in range(5):
                if (1 << v) & adj[c][u]:
                    c_old = c
                    break
            
            if c_old == c_new:
                return 0
            
            # e_old: C4s using (u,v) in c_old.
            # (u,v) is currently in adj[c_old].
            e_old = count_c4_through_edge(u, v, c_old)
            
            # e_new: C4s using (u,v) in c_new.
            # (u,v) is currently NOT in adj[c_new].
            e_new = count_c4_through_edge(u, v, c_new)
            
            return e_new - e_old

        def apply_change(u, v, c_new):
            # Returns delta
            c_old = -1
            for c in range(5):
                if (1 << v) & adj[c][u]:
                    c_old = c
                    break
            if c_old == c_new:
                return 0
            
            d = delta_energy(u, v, c_new)
            
            # Execute change
            adj[c_old][u] &= ~(1 << v)
            adj[c_old][v] &= ~(1 << u)
            adj[c_new][u] |= (1 << v)
            adj[c_new][v] |= (1 << u)
            
            return d

        # Initialize
        for i in range(N):
            for j in range(i+1, N):
                c = random.randint(0, 4)
                adj[c][i] |= (1 << j)
                adj[c][j] |= (1 << i)
        
        def total_energy():
            total = 0
            for i in range(N):
                for j in range(i+1, N):
                    # Find color
                    for c in range(5):
                        if (1 << j) & adj[c][i]:
                            total += count_c4_through_edge(i, j, c)
                            break
            return total // 4

        current_energy = total_energy()
        
        # Local Search Loop
        max_iter = 500000
        temperature = 1.0
        cooling = 0.999
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            
            if current_energy == 0:
                # Success!
                final_colors = []
                for i in range(N):
                    for j in range(i+1, N):
                        for c in range(5):
                            if (1 << j) & adj[c][i]:
                                final_colors.append(c)
                                break
                return {'n': N, 'colors': final_colors}
            
            # Pick an edge
            # Strategy: If energy > 0, pick an edge in a C4.
            if current_energy > 0:
                # Pick random edge until we find one in a C4?
                # Or just pick random edge and check.
                # To be efficient, we can pick a random vertex u, then random v.
                # Check if (u,v) is in a C4.
                # If yes, change it.
                # If no, try a few times, then fallback to random.
                
                found_c4_edge = False
                for _attempt in range(5):
                    u = random.randint(0, N-1)
                    v = random.randint(0, N-1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    
                    # Find color
                    c_old = -1
                    for c in range(5):
                        if (1 << v) & adj[c][u]:
                            c_old = c
                            break
                    
                    if count_c4_through_edge(u, v, c_old) > 0:
                        found_c4_edge = True
                        break
                
                if not found_c4_edge:
                    u = random.randint(0, N-1)
                    v = random.randint(0, N-1)
                    if u == v:
                        continue
                    if u > v:
                        u, v = v, u
                    c_old = -1
                    for c in range(5):
                        if (1 << v) & adj[c][u]:
                            c_old = c
                            break

            else:
                u = random.randint(0, N-1)
                v = random.randint(0, N-1)
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                c_old = -1
                for c in range(5):
                    if (1 << v) & adj[c][u]:
                        c_old = c
                        break

            # Try all other colors
            best_delta = 0
            best_c = -1
            
            for c_new in range(5):
                if c_new == c_old:
                    continue
                d = delta_energy(u, v, c_new)
                if d < best_delta:
                    best_delta = d
                    best_c = c_new
            
            if best_c != -1:
                apply_change(u, v, best_c)
                current_energy += best_delta
            else:
                # Accept worse move with probability
                c_rand = random.randint(0, 4)
                if c_rand != c_old:
                    d = delta_energy(u, v, c_rand)
                    if d < 0 or random.random() < pow(2.718, -d / temperature):
                        apply_change(u, v, c_rand)
                        current_energy += d
                temperature *= cooling

        return None

    # Try N from 29 down to 26
    # We know 26 is possible.
    total_time_limit = 110
    start_total = time.time()
    
    for N in [29, 28, 27, 26]:
        time_remaining = total_time_limit - (time.time() - start_total)
        if time_remaining < 5:
            break
        
        # Allocate more time for larger N if we think it's possible?
        # Or just give each a chunk.
        # 29 and 28 are very hard. 27 is hard. 26 is easy.
        # Let's give 26 the most time if others fail?
        # Actually, if we find 26, we can stop.
        # But we want to maximize n.
        # So try 29, then 28, then 27, then 26.
        
        alloc_time = min(10, time_remaining)
        
        res = solve_local_search(N, time_limit=alloc_time)
        if res is not None:
            return res
            
    # Fallback to seed solution if all fail (should not happen for 26)
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
