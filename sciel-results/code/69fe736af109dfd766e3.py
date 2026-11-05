import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
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

    def solve_n(n, time_limit):
        # Indices for edges
        # i < j
        # idx = i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        # Or simpler: list of edges
        edges = []
        edge_index = {}
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_index[(i, j)] = k
                edges.append((i, j))
                k += 1
        
        # Initialize colors randomly
        colors = [random.randint(0, 3) for _ in range(len(edges))]
        
        # Initialize adjacency bitsets
        # adj[c][u] is a bitset of neighbors of u with color c
        adj = [[0] * n for _ in range(4)]
        
        for (u, v), c in zip(edges, colors):
            adj[c][u] |= (1 << v)
            adj[c][v] |= (1 << u)
            
        # Calculate initial total monochromatic triangles
        total_mono = 0
        # For each color c, count triangles
        # A triangle (u,v,w) is mono if adj[c][u] has v and w, adj[c][v] has u and w, etc.
        # Sum over u < v < w: 1 if adj[c][u] & (1<<v) and adj[c][u] & (1<<w) and adj[c][v] & (1<<w)
        # This is O(n^3 * 4) which is 50^3 * 4 = 500,000 ops. Acceptable.
        # Faster: For each u, v, count common neighbors w > v.
        # But we can just compute it once.
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[c][i] & (1 << j):
                        # Count common neighbors k > j
                        common = adj[c][i] & adj[c][j]
                        # Mask to only include k > j
                        # Actually, we just need to count bits set in common that are > j
                        # Or just sum bit_count(common) for all i<j and divide by 3?
                        # No, each triangle is counted 3 times (once for each vertex as 'i').
                        # So total triangles for color c = sum_{i<j, edge(i,j) in c} bit_count(adj[c][i] & adj[c][j]) / 3
                        # Wait, if we sum for all i<j, each triangle (u,v,w) is counted:
                        # Pair (u,v): w is in common.
                        # Pair (u,w): v is in common.
                        # Pair (v,w): u is in common.
                        # So yes, sum / 3.
                        total_mono += (adj[c][i] & adj[c][j]).bit_count()
        total_mono //= 3
        
        if total_mono == 0:
            return {'n': n, 'colors': colors}

        # Local Search
        temp = 10.0
        cooling = 0.999
        best_total = total_mono
        best_colors = colors[:]
        
        while time.time() < start_time + time_limit:
            # Pick random edge
            idx = random.randint(0, len(edges) - 1)
            u, v = edges[idx]
            c_old = colors[idx]
            
            # Find best new color
            best_delta = 0
            best_c_new = c_old
            for c_new in range(4):
                if c_new == c_old:
                    continue
                # Calculate delta
                # N_old = bit_count(adj[c_old][u] & adj[c_old][v])
                # N_new = bit_count(adj[c_new][u] & adj[c_new][v])
                # Note: adj[c_old][u] and adj[c_old][v] currently include each other?
                # Yes, adj[c][u] has bit v if edge(u,v) is color c.
                # When calculating common neighbors for triangle (u,v,w), w must be != u, v.
                # The bit v in adj[c_old][u] is set. The bit u in adj[c_old][v] is set.
                # Intersection adj[c_old][u] & adj[c_old][v] will have bit v set?
                # adj[c_old][u] has bit v. adj[c_old][v] has bit u.
                # Bit v in adj[c_old][v] is 0 (no self loop).
                # So bit v is NOT in the intersection.
                # Similarly bit u is NOT in the intersection.
                # So the intersection correctly counts w != u, v.
                
                n_old = (adj[c_old][u] & adj[c_old][v]).bit_count()
                n_new = (adj[c_new][u] & adj[c_new][v]).bit_count()
                delta = n_new - n_old
                
                if delta < best_delta:
                    best_delta = delta
                    best_c_new = c_new
            
            # Accept if best_delta < 0 or with probability exp(-best_delta / temp)
            accept = False
            if best_delta < 0:
                accept = True
            else:
                if random.random() < pow(2.71828, -best_delta / temp):
                    accept = True
            
            if accept:
                # Update state
                colors[idx] = best_c_new
                adj[c_old][u] ^= (1 << v)
                adj[c_old][v] ^= (1 << u)
                adj[best_c_new][u] ^= (1 << v)
                adj[best_c_new][v] ^= (1 << u)
                total_mono += best_delta
                temp *= cooling
                
                if total_mono < best_total:
                    best_total = total_mono
                    best_colors = colors[:]
                    if total_mono == 0:
                        return {'n': n, 'colors': colors}
            else:
                temp *= cooling
                
        if best_total == 0:
            return {'n': n, 'colors': best_colors}
        return None

    # Try to find solutions for n=50 down to 33
    # Allocate time wisely
    time_per_n = 8.0
    for n in range(50, 32, -1):
        if time.time() > deadline - 5:
            break
        remaining = deadline - time.time() - 2
        if remaining < 1:
            break
        t_limit = min(time_per_n, remaining)
        res = solve_n(n, t_limit)
        if res:
            return res
            
    return get_fallback()
