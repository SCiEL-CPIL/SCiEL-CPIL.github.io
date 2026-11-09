import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115.0
    
    # Fallback: The known n=32 construction
    def get_seed_32():
        def mul(a, b):
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
            x = mul(x, 2)
            
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return 32, colors

    best_n, best_colors = get_seed_32()
    
    def try_solve(n, time_limit):
        N = n*(n-1)//2
        edges = []
        edge_to_idx = {}
        for i in range(n):
            for j in range(i+1, n):
                edge_to_idx[(i, j)] = len(edges)
                edges.append((i, j))
        
        # Precompute triangle neighbors for each edge as numpy arrays for speed
        # tri_neighbors[e] = array of shape (n-2, 2)
        tri_neighbors = [None] * N
        for idx, (i, j) in enumerate(edges):
            neighbors = []
            for w in range(n):
                if w == i or w == j: continue
                idx_iw = edge_to_idx.get((i, w) if i < w else (w, i))
                idx_jw = edge_to_idx.get((j, w) if j < w else (w, j))
                neighbors.append((idx_iw, idx_jw))
            tri_neighbors[idx] = np.array(neighbors, dtype=np.int32)
        
        best_colors = None
        best_count = float('inf')
        
        start_time = time.time()
        
        # Multiple restarts
        while time.time() < start_time + time_limit:
            # Greedy Init: Pick edges in random order, assign color with min immediate conflict
            colors = np.zeros(N, dtype=np.int8)
            edge_order = np.random.permutation(N)
            
            # To make greedy init fast, we might just do random first, then polish?
            # Greedy init is O(N * (n-2) * 4). For N=1225, n=50, that's ~250k ops. Fast enough.
            # But we need to track counts.
            # Actually, simple random init + fast SA is often competitive.
            # Let's stick to random init but optimize the SA loop.
            colors = np.random.randint(0, 4, size=N, dtype=np.int8)
            
            # Calculate initial triangle count
            curr_count = 0
            for e in range(N):
                c_e = colors[e]
                if len(tri_neighbors[e]) == 0: continue
                neigh = tri_neighbors[e]
                c1 = colors[neigh[:, 0]]
                c2 = colors[neigh[:, 1]]
                # Count where c1 == c_e and c2 == c_e
                curr_count += np.sum((c1 == c_e) & (c2 == c_e))
            curr_count //= 3
            
            if curr_count == 0:
                return colors.tolist()
            
            if curr_count < best_count:
                best_count = curr_count
                best_colors = colors.copy()
            
            # Simulated Annealing
            T = 1.0
            T_min = 0.01
            cooling = 0.999
            no_improve_steps = 0
            max_no_improve = 5000 # Increased to allow more exploration
            
            while time.time() < start_time + time_limit and no_improve_steps < max_no_improve:
                e = np.random.randint(0, N)
                old_c = colors[e]
                
                # Get neighbors
                neigh = tri_neighbors[e]
                if len(neigh) == 0:
                    # No triangles, any color is fine, keep old
                    T *= cooling
                    if T < T_min: T = T_min
                    continue
                    
                c1 = colors[neigh[:, 0]]
                c2 = colors[neigh[:, 1]]
                
                # Calculate delta for each of the 4 colors
                # delta[c] = count(new_c == c1 & c2) - count(old_c == c1 & c2)
                # We only care about new_c != old_c
                
                # Vectorized calculation
                # For each color k in 0..3:
                # matches_k = np.sum((c1 == k) & (c2 == k))
                # delta_k = matches_k - matches_old
                
                matches = np.array([np.sum((c1 == k) & (c2 == k)) for k in range(4)])
                matches_old = matches[old_c]
                
                best_new_c = -1
                best_delta = 0
                best_new_count = curr_count
                
                for new_c in range(4):
                    if new_c == old_c: continue
                    delta = matches[new_c] - matches_old
                    new_count = curr_count + delta
                    if new_count < best_new_count:
                        best_new_count = new_count
                        best_new_c = new_c
                        best_delta = delta
                        
                if best_new_c != -1:
                    # Accept if improving or with probability exp(-delta/T)
                    if best_delta < 0:
                        accept = True
                    else:
                        accept = random.random() < np.exp(-best_delta / T)
                    
                    if accept:
                        colors[e] = best_new_c
                        curr_count = best_new_count
                        
                        if curr_count < best_count:
                            best_count = curr_count
                            best_colors = colors.copy()
                            no_improve_steps = 0
                        else:
                            no_improve_steps += 1
                            
                        if curr_count == 0:
                            return colors.tolist()
                    else:
                        no_improve_steps += 1
                        
                T *= cooling
                if T < T_min:
                    T = T_min
                    
        return best_colors.tolist() if best_colors is not None else None

    # Try n from 50 down to 33
    candidates = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    time_budgets = {
        50: 40.0,
        49: 20.0,
        48: 15.0,
        47: 10.0,
        46: 5.0,
        45: 5.0,
        44: 3.0,
        43: 2.0,
        42: 2.0,
        41: 1.0,
        40: 1.0,
        39: 1.0,
        38: 1.0,
        37: 1.0,
        36: 1.0,
        35: 1.0,
        34: 1.0,
        33: 1.0
    }
    
    for n in candidates:
        if time.time() > deadline - 2.0:
            break
            
        t_limit = min(time_budgets.get(n, 1.0), deadline - time.time() - 1.0)
        if t_limit <= 0:
            continue
            
        result = try_solve(n, t_limit)
        if result is not None:
            # Verify result
            N = n*(n-1)//2
            edges = []
            edge_to_idx = {}
            for i in range(n):
                for j in range(i+1, n):
                    edge_to_idx[(i, j)] = len(edges)
                    edges.append((i, j))
            
            colors = np.array(result)
            tri_count = 0
            for idx, (i, j) in enumerate(edges):
                c_ij = colors[idx]
                for w in range(n):
                    if w == i or w == j: continue
                    idx_iw = edge_to_idx.get((i, w) if i < w else (w, i))
                    idx_jw = edge_to_idx.get((j, w) if j < w else (w, j))
                    if colors[idx_iw] == c_ij and colors[idx_jw] == c_ij:
                        tri_count += 1
            tri_count //= 3
            
            if tri_count == 0:
                best_n = n
                best_colors = result
                return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}
