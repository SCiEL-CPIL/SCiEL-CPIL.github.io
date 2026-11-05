import time
import random
import sys

DEADLINE = 145  # seconds, safety margin before 150s kill

def _time_left():
    return time.time() - _START_TIME < DEADLINE

_START_TIME = time.time()

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    # ---- Build the n=32 seed colouring (Greenwood-Gleason doubled) ----
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

    n_seed = 32
    colors_seed = []
    for i in range(n_seed):
        for j in range(i + 1, n_seed):
            if i // 16 == j // 16:
                colors_seed.append(log[(i % 16) ^ (j % 16)] % 3)
            else:
                colors_seed.append(3)

    def _validate(colors, n):
        """Check no monochromatic triangle."""
        # Efficient check using bitmasks
        adj = [set() for _ in range(4)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj[c].add((i, j))
                idx += 1
        for c in range(4):
            edges = adj[c]
            if not edges:
                continue
            # Build adjacency list for this color
            adj_list = {i: set() for i in range(n)}
            for (i, j) in edges:
                adj_list[i].add(j)
                adj_list[j].add(i)
            for i in range(n):
                for j in adj_list[i]:
                    if j > i:
                        # Check if i-j-k is a triangle in color c
                        # Common neighbors of i and j in color c
                        common = adj_list[i] & adj_list[j]
                        if common:
                            return False
        return True

    if not _validate(colors_seed, n_seed):
        return {'n': 1, 'colors': []}

    best_n = n_seed
    best_colors = colors_seed[:]

    # ---- Attempt to find a larger coloring using Random Search + Local Search ----
    # Target: n=50 (Chung's record)
    # If 50 fails, try 45, 40, etc.
    
    def try_find_coloring(n_target, time_limit=100):
        """
        Attempt to find a valid 4-coloring of K_n_target.
        Returns list of colors if found, else None.
        """
        start_time = time.time()
        
        # Edge index calculation
        # We store colors in a flat list of length n*(n-1)//2
        # index(i, j) for i < j
        # Precompute indices for speed? Or just use a 2D array for lookup and update?
        # For n=50, 2500 entries in 2D array is fine.
        
        def get_idx(i, j):
            if i > j:
                i, j = j, i
            # Formula for row-major lex order of pairs (i,j) with i<j
            # Number of pairs starting with k < i is sum_{k=0}^{i-1} (n-1-k)
            # = i*(n-1) - i*(i-1)//2
            # Then j is the (j-i-1)-th pair starting with i
            return i * (n_target - 1) - i * (i - 1) // 2 + (j - i - 1)

        # Initialize random coloring
        num_edges = n_target * (n_target - 1) // 2
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Data structures for local search
        # neighbors[c][u] is a bitmask of vertices v such that edge (u,v) has color c
        # Note: edge (u,v) is undirected. We store u in N_c(v) and v in N_c(u).
        neighbors = [[0] * n_target for _ in range(4)]
        
        # Initialize neighbors
        for i in range(n_target):
            for j in range(i + 1, n_target):
                c = colors[get_idx(i, j)]
                neighbors[c][i] |= (1 << j)
                neighbors[c][j] |= (1 << i)
                
        # Calculate initial total number of monochromatic triangles
        # A triangle (u,v,w) is monochromatic if all 3 edges have the same color.
        # We can count by summing over all edges (u,v) of color c, the number of common neighbors in N_c.
        # But each triangle is counted 3 times (once per edge).
        # So Total Triangles = (1/3) * sum_{u<v} popcount(N_{c(u,v)}(u) & N_{c(u,v)}(v))
        
        def count_triangles():
            total = 0
            for i in range(n_target):
                for j in range(i + 1, n_target):
                    c = colors[get_idx(i, j)]
                    total += bin(neighbors[c][i] & neighbors[c][j]).count('1')
            return total // 3
            
        current_conflicts = count_triangles()
        
        # Local Search
        # We will try to reduce current_conflicts to 0.
        # Move: change color of edge (u,v) from c_old to c_new.
        # Delta conflicts = (new_conflicts) - (old_conflicts)
        
        # Optimization: Precompute list of edges
        edges = []
        for i in range(n_target):
            for j in range(i + 1, n_target):
                edges.append((i, j))
                
        # Shuffle edges to avoid bias
        random.shuffle(edges)
        
        no_improvement_count = 0
        max_no_improvement = 10000 # Restart if stuck
        
        while current_conflicts > 0 and time.time() - start_time < time_limit:
            if not _time_left():
                return None
                
            # Try a random edge
            u, v = random.choice(edges)
            idx = get_idx(u, v)
            c_old = colors[idx]
            
            # Try all 3 other colors
            best_delta = 10**9
            best_c_new = -1
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                
                # Calculate delta
                # Old contribution of edge (u,v) to conflict count:
                # popcount(N_{c_old}(u) & N_{c_old}(v))
                old_contrib = bin(neighbors[c_old][u] & neighbors[c_old][v]).count('1')
                
                # New contribution if we change to c_new:
                # The sets N_{c_new}(u) and N_{c_new}(v) will include each other?
                # Currently, u is NOT in N_{c_new}(v) and v is NOT in N_{c_new}(u) because edge is c_old.
                # After change, v will be in N_{c_new}(u) and u will be in N_{c_new}(v).
                # The intersection N'_{c_new}(u) & N'_{c_new}(v) will be:
                # (N_{c_new}(u) | {v}) & (N_{c_new}(v) | {u})
                # = (N_{c_new}(u) & N_{c_new}(v)) | (N_{c_new}(u) & {u}) | ({v} & N_{c_new}(v)) | ({v} & {u})
                # u is not in N_{c_new}(u) by definition (no self loops). So N_{c_new}(u) & {u} = 0.
                # v is not in N_{c_new}(v) by definition. So {v} & N_{c_new}(v) = 0.
                # {v} & {u} = 0.
                # So the new contribution is just popcount(N_{c_new}(u) & N_{c_new}(v)).
                
                new_contrib = bin(neighbors[c_new][u] & neighbors[c_new][v]).count('1')
                
                delta = new_contrib - old_contrib
                
                if delta < best_delta:
                    best_delta = delta
                    best_c_new = c_new
            
            if best_c_new == -1:
                continue
                
            if best_delta < 0:
                # Accept move
                colors[idx] = best_c_new
                current_conflicts += best_delta
                no_improvement_count = 0
                
                # Update neighbor sets
                # Remove u from N_{c_old}(v) and v from N_{c_old}(u)
                neighbors[c_old][u] &= ~(1 << v)
                neighbors[c_old][v] &= ~(1 << u)
                
                # Add u to N_{best_c_new}(v) and v to N_{best_c_new}(u)
                neighbors[best_c_new][u] |= (1 << v)
                neighbors[best_c_new][v] |= (1 << u)
                
            else:
                # No improvement with this edge
                no_improvement_count += 1
                
            if no_improvement_count > max_no_improvement:
                # Random restart
                colors = [random.randint(0, 3) for _ in range(num_edges)]
                for i in range(n_target):
                    for c in range(4):
                        neighbors[c][i] = 0
                for i in range(n_target):
                    for j in range(i + 1, n_target):
                        c = colors[get_idx(i, j)]
                        neighbors[c][i] |= (1 << j)
                        neighbors[c][j] |= (1 << i)
                current_conflicts = count_triangles()
                no_improvement_count = 0
                random.shuffle(edges)
                
        if current_conflicts == 0:
            return colors
        else:
            return None

    # Try n=50 first
    # We have ~140 seconds total. Seed construction is fast.
    # Let's give 100s to n=50.
    if _time_left() and time.time() - _START_TIME < 100:
        res_50 = try_find_coloring(50, time_limit=95)
        if res_50:
            return {'n': 50, 'colors': res_50}
            
    # If 50 failed, try 45
    if _time_left() and time.time() - _START_TIME < 120:
        res_45 = try_find_coloring(45, time_limit=20)
        if res_45:
            return {'n': 45, 'colors': res_45}

    # If 45 failed, try 40
    if _time_left() and time.time() - _START_TIME < 140:
        res_40 = try_find_coloring(40, time_limit=15)
        if res_40:
            return {'n': 40, 'colors': res_40}
            
    # Fallback to seed n=32
    return {'n': best_n, 'colors': best_colors}

