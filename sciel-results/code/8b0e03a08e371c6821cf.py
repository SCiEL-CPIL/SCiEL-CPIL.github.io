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
    
    def get_idx(i, j, n):
        if i > j: i, j = j, i
        return i*n - i*(i+1)//2 + (j-i-1)

    def count_triangles(colors, n):
        count = 0
        # Vectorized check is hard for arbitrary n without precomputed lists
        # We'll use the precomputed list approach in the solver
        return 0 # Placeholder, solver will calculate

    def try_solve(n, time_limit):
        N = n*(n-1)//2
        edges = []
        edge_to_idx = {}
        for i in range(n):
            for j in range(i+1, n):
                edge_to_idx[(i, j)] = len(edges)
                edges.append((i, j))
        
        # Precompute triangle neighbors for each edge
        # tri_neighbors[e_idx] = list of (idx_uw, idx_vw)
        tri_neighbors = [[] for _ in range(N)]
        for idx, (i, j) in enumerate(edges):
            for w in range(n):
                if w == i or w == j: continue
                idx_iw = edge_to_idx.get((i, w) if i < w else (w, i))
                idx_jw = edge_to_idx.get((j, w) if j < w else (w, j))
                tri_neighbors[idx].append((idx_iw, idx_jw))
        
        # Convert to numpy arrays for vectorization
        # For each edge e, we have a list of pairs (e1, e2).
        # We can store e1s and e2s as numpy arrays.
        tri_e1 = [None] * N
        tri_e2 = [None] * N
        for idx in range(N):
            if tri_neighbors[idx]:
                e1s, e2s = zip(*tri_neighbors[idx])
                tri_e1[idx] = np.array(e1s, dtype=np.int32)
                tri_e2[idx] = np.array(e2s, dtype=np.int32)
            else:
                tri_e1[idx] = np.array([], dtype=np.int32)
                tri_e2[idx] = np.array([], dtype=np.int32)

        best_colors = None
        best_count = float('inf')
        
        start_time = time.time()
        
        # Multiple restarts
        while time.time() < start_time + time_limit:
            # Random init
            colors = np.random.randint(0, 4, size=N)
            
            # Calculate initial triangle count
            curr_count = 0
            for e in range(N):
                c_e = colors[e]
                e1s = tri_e1[e]
                e2s = tri_e2[e]
                if len(e1s) > 0:
                    mask = (colors[e1s] == c_e) & (colors[e2s] == c_e)
                    curr_count += np.sum(mask)
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
            max_no_improve = 5000
            
            while time.time() < start_time + time_limit and no_improve_steps < max_no_improve:
                e = np.random.randint(0, N)
                old_c = colors[e]
                
                # Try other 3 colors
                best_new_c = -1
                best_delta = 0
                best_new_count = curr_count
                
                e1s = tri_e1[e]
                e2s = tri_e2[e]
                
                if len(e1s) == 0:
                    new_c = random.randint(0, 3)
                    colors[e] = new_c
                    continue

                c1s = colors[e1s]
                c2s = colors[e2s]
                
                old_mask = (old_c == c1s) & (old_c == c2s)
                old_contrib = np.sum(old_mask)
                
                for new_c in range(4):
                    if new_c == old_c: continue
                    
                    new_mask = (new_c == c1s) & (new_c == c2s)
                    new_contrib = np.sum(new_mask)
                    
                    delta = new_contrib - old_contrib
                    new_count = curr_count + delta
                    
                    if new_count < best_new_count:
                        best_new_count = new_count
                        best_new_c = new_c
                        best_delta = delta
                        
                if best_new_c != -1:
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
    # We prioritize higher n.
    # Give each n a fraction of the time.
    # If we find a solution for n, we update best and continue to higher n?
    # The task says "Maximize n".
    # So we should try the largest n first?
    # If we fail at 50, we might still find 49.
    # Let's try 50, 49, ..., 33.
    # Allocate time based on remaining time.
    
    # It is better to try the highest possible n first to maximize the score.
    # If we find n=50, we are done (or can try 51 which is impossible).
    # If we don't find 50, we try 49, etc.
    
    # Let's reserve 5 seconds for the final return and verification.
    # We will try n=50 with 30s, n=49 with 20s, etc.
    # Or just split remaining time evenly among candidates.
    
    candidates = [50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33]
    
    # We will try them in order.
    # If we find a valid coloring for n, we store it.
    # We don't stop immediately if we find n=40, because we might find n=50 later?
    # No, if we are iterating 50 down to 33, the first one we find is the max.
    # BUT, finding a solution for 50 is harder than 40.
    # So we should probably give more time to 50.
    
    # Strategy:
    # 1. Try n=50 for 40s. If found, return.
    # 2. Try n=49 for 20s. If found, save.
    # 3. Try n=48 for 15s. If found, save.
    # ...
    # 4. Return best found.
    
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
            # Re-check triangle count for safety
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
                # Since we are iterating from high to low, this is the best so far.
                # We can continue to see if we can find a higher n? No, we started at 50.
                # So if we found 50, we are done.
                # If we found 49, we might have failed at 50.
                # We should stop if we find a solution?
                # Yes, because we are trying in descending order.
                # Wait, if we fail at 50, we try 49. If we succeed at 49, is it possible we could have found 50 with more time?
                # Maybe. But we allocated specific budgets.
                # To be safe and maximize score, if we find ANY valid n, we should probably keep searching for higher n if we have time?
                # No, the loop is descending. So if we find 49, we already failed at 50.
                # So 49 is the best we got from that budget.
                # We should return immediately?
                # If we return immediately, we might miss a chance to find 50 if we hadn't tried it yet?
                # We are trying 50 first. So if we are at 49, we already tried 50.
                # So returning the first valid found in descending order is optimal.
                return {'n': best_n, 'colors': best_colors}
            
    return {'n': best_n, 'colors': best_colors}
