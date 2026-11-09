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
    
    # Helper to count triangles
    def count_triangles(colors, n):
        count = 0
        # Precompute edge indices for speed? 
        # For n=33, N=528. 
        # We can just iterate.
        for i in range(n):
            for j in range(i+1, n):
                idx_ij = i*n - i*(i+1)//2 + (j-i-1)
                c_ij = colors[idx_ij]
                for k in range(j+1, n):
                    idx_ik = i*n - i*(i+1)//2 + (k-i-1)
                    idx_jk = j*n - j*(j+1)//2 + (k-j-1)
                    if colors[idx_ik] == c_ij and colors[idx_jk] == c_ij:
                        count += 1
        return count

    def get_idx(i, j, n):
        if i > j: i, j = j, i
        return i*n - i*(i+1)//2 + (j-i-1)

    def try_solve(n, time_limit):
        N = n*(n-1)//2
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                edges.append((i, j))
        
        # Precompute neighbors for each edge
        # edge_neighbors[e_idx] = list of (idx_uw, idx_vw) for triangles (u,v,w)
        edge_neighbors = []
        for idx, (i, j) in enumerate(edges):
            neighbors = []
            for w in range(n):
                if w == i or w == j: continue
                idx_iw = get_idx(i, w, n)
                idx_jw = get_idx(j, w, n)
                neighbors.append((idx_iw, idx_jw))
            edge_neighbors.append(neighbors)
            
        # Random init
        colors = [random.randint(0, 3) for _ in range(N)]
        curr_count = count_triangles(colors, n)
        
        if curr_count == 0:
            return colors
            
        start_time = time.time()
        T = 50.0 # Initial temperature
        T_min = 0.1
        cooling_rate = 0.995
        
        while time.time() < start_time + time_limit:
            # Pick random edge
            e_idx = random.randint(0, N-1)
            old_c = colors[e_idx]
            
            # Try other 3 colors
            best_new_c = -1
            best_new_count = curr_count
            best_delta = 0
            
            for new_c in range(4):
                if new_c == old_c: continue
                
                delta = 0
                for idx_iw, idx_jw in edge_neighbors[e_idx]:
                    c_iw = colors[idx_iw]
                    c_jw = colors[idx_jw]
                    
                    # Before: mono if old_c == c_iw == c_jw
                    if old_c == c_iw and old_c == c_jw:
                        delta -= 1
                    # After: mono if new_c == c_iw == c_jw
                    if new_c == c_iw and new_c == c_jw:
                        delta += 1
                        
                new_count = curr_count + delta
                if new_count < best_new_count:
                    best_new_count = new_count
                    best_new_c = new_c
                    best_delta = delta
                    
            if best_new_c != -1:
                # Acceptance criterion: Simulated Annealing
                if best_delta < 0 or random.random() < np.exp(-best_delta / max(T, 1e-9)):
                    colors[e_idx] = best_new_c
                    curr_count = best_new_count
                    
                    if curr_count == 0:
                        return colors
            T *= cooling_rate
            if T < T_min:
                T = T_min
                
        return None

    # Main loop
    # Try to find solutions for n > 32
    # We can try n=33, 34, ...
    # If we find one, we update best and continue to higher n?
    # Or do we try to find the max n?
    # The task says "Maximize n".
    # So we should try the largest n first? Or smallest?
    # If we find n=50, we are done (or try 51).
    # If we don't find 50, maybe 49?
    # Given the difficulty, starting from 33 and going up is safer to ensure progress.
    # But if 33 is easy, we might waste time.
    # Let's try 33, then 34, etc.
    
    for n in range(33, 63):
        if time.time() > deadline - 5:
            break
            
        # Give each n a time slice
        # If we are at n=33, we have lots of time.
        # If we are at n=50, we have less.
        # Let's allocate time based on remaining time.
        remaining = deadline - time.time()
        # Heuristic: give 5-10 seconds per n
        t_limit = min(10.0, remaining * 0.1)
        
        result = try_solve(n, t_limit)
        if result is not None:
            best_n = n
            best_colors = result
            # Continue to try higher n?
            # Yes, because we want to maximize n.
        else:
            # If we fail for n, it doesn't mean n+1 is impossible.
            # But it might be a sign that the search is not powerful enough.
            # We continue anyway.
            pass

    return {'n': best_n, 'colors': best_colors}
