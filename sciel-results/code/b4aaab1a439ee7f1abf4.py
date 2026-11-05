import time
import random

_START_TIME = time.time()
DEADLINE = 110

def _time_left():
    return time.time() - _START_TIME < DEADLINE

def construct():
    global _START_TIME
    _START_TIME = time.time()
    random.seed(42)

    # Target n values to try
    target_ns = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32]
    
    best_n = 32
    best_colors = None
    
    # Precompute a known valid 32-coloring as fallback
    # Using the Greenwood-Gleason construction from the seed
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
                
    # Validation function
    def _validate(colors, n):
        # Check no monochromatic triangle
        # Build adjacency lists for each color
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
            adj_list = {i: set() for i in range(n)}
            for (i, j) in edges:
                adj_list[i].add(j)
                adj_list[j].add(i)
            for i in range(n):
                for j in adj_list[i]:
                    if j > i:
                        if adj_list[i] & adj_list[j]:
                            return False
        return True

    if _validate(colors_seed, n_seed):
        best_n = n_seed
        best_colors = colors_seed[:]
    else:
        return {'n': 1, 'colors': []}

    # Local Search for larger n
    for n in target_ns:
        if not _time_left():
            break
            
        # If we already have a better solution, skip
        if n <= best_n:
            break
            
        # Try to find a valid coloring for this n
        found = False
        # Number of restarts
        max_restarts = 20 if n > 45 else 50
        
        for restart in range(max_restarts):
            if not _time_left():
                break
                
            # Initialize random coloring
            num_edges = n * (n - 1) // 2
            colors = [random.randint(0, 3) for _ in range(num_edges)]
            
            # Precompute edge indices for fast access?
            # edge_idx[i][j] -> index in colors
            # Or just a function
            def get_idx(i, j):
                if i > j: i, j = j, i
                return i * n - i * (i + 1) // 2 + (j - i - 1)
            
            # Calculate initial cost
            def calc_cost(colors):
                cost = 0
                # For each triangle, check if mono
                # This is O(N^3) which is slow for N=50 (125k triangles)
                # But we can optimize by iterating over edges and checking common neighbors?
                # Or just simple triple loop for initialization, then incremental for search.
                # Simple triple loop:
                for i in range(n):
                    for j in range(i + 1, n):
                        c_ij = colors[get_idx(i, j)]
                        for k in range(j + 1, n):
                            c_jk = colors[get_idx(j, k)]
                            c_ik = colors[get_idx(i, k)]
                            if c_ij == c_jk == c_ik:
                                cost += 1
                return cost

            current_cost = calc_cost(colors)
            
            if current_cost == 0:
                if _validate(colors, n):
                    best_n = n
                    best_colors = colors[:]
                    found = True
                    break

            # Local Search
            # To speed up, we can precompute for each edge, the list of triangles it belongs to.
            # But dynamic update is O(N).
            
            stuck_count = 0
            max_stuck = 1000
            
            while _time_left() and current_cost > 0:
                if stuck_count > max_stuck:
                    break
                
                # Pick random edge
                e_idx = random.randint(0, num_edges - 1)
                # Find i, j for e_idx
                # Inverse of get_idx
                # e_idx = i*n - i(i+1)/2 + (j-i-1)
                # This is hard to invert quickly.
                # Let's store the mapping.
                # Actually, we can just iterate over all edges and pick one? No.
                # Let's create a list of (i, j, idx) pairs.
                
                # Optimization: Precompute edge list
                edge_list = [(i, j, i * n - i * (i + 1) // 2 + (j - i - 1)) for i in range(n) for j in range(i + 1, n)]
                
                # Pick random edge from list
                i, j, idx = edge_list[e_idx]
                
                old_c = colors[idx]
                new_c = random.randint(0, 3)
                if new_c == old_c:
                    continue
                    
                # Calculate delta cost
                # Triangles containing (i, j) are (i, j, k) for k != i, j
                delta = 0
                for k in range(n):
                    if k == i or k == j:
                        continue
                    
                    # Get colors of (i,k) and (j,k)
                    ik_idx = i * n - i * (i + 1) // 2 + (k - i - 1) if i < k else k * n - k * (k + 1) // 2 + (i - k - 1)
                    jk_idx = j * n - j * (j + 1) // 2 + (k - j - 1) if j < k else k * n - k * (k + 1) // 2 + (j - k - 1)
                    
                    c_ik = colors[ik_idx]
                    c_jk = colors[jk_idx]
                    
                    # Old triangle (i,j,k)
                    if old_c == c_ik == c_jk:
                        delta -= 1
                    # New triangle (i,j,k)
                    if new_c == c_ik == c_jk:
                        delta += 1

                if delta < 0:
                    colors[idx] = new_c
                    current_cost += delta
                    stuck_count = 0
                else:
                    # Accept with small probability
                    if random.random() < 0.05:
                        colors[idx] = new_c
                        current_cost += delta
                    else:
                        stuck_count += 1

        if found:
            break

    return {'n': best_n, 'colors': best_colors}
