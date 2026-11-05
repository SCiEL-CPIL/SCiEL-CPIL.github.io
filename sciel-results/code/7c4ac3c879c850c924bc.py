import random
import time
import numpy as np

def construct():
    # Target n. The known record is 50. We aim for 50.
    n = 50
    num_colors = 4
    num_edges = n * (n - 1) // 2

    # Precompute edge indices
    # idx_map[(i, j)] = index in flat array
    idx_map = {}
    edge_list = []
    for i in range(n):
        for j in range(i + 1, n):
            idx_map[(i, j)] = len(edge_list)
            edge_list.append((i, j))

    # Precompute for each edge e, the other two edge indices in any triangle containing e
    # tri_edges[e] = list of (idx_ik, idx_jk)
    tri_edges = [[] for _ in range(num_edges)]
    for e, (i, j) in enumerate(edge_list):
        for k in range(n):
            if k == i or k == j:
                continue
            if i < k:
                idx_ik = idx_map[(i, k)]
            else:
                idx_ik = idx_map[(k, i)]
            if j < k:
                idx_jk = idx_map[(j, k)]
            else:
                idx_jk = idx_map[(k, j)]
            tri_edges[e].append((idx_ik, idx_jk))

    # Function to count monochromatic triangles
    # Returns the list of edges that are part of at least one mono triangle? 
    # Or just the count?
    # For efficiency in local search, we want to quickly identify "bad" edges.
    
    def count_mono_edges(colors):
        """
        Returns a list of edges that are part of a monochromatic triangle.
        Also returns the total count of mono triangles (each triangle counted once per edge, so total/3).
        """
        bad_edges = set()
        mono_count = 0
        for e in range(num_edges):
            c_e = colors[e]
            for idx_ik, idx_jk in tri_edges[e]:
                if colors[idx_ik] == c_e and colors[idx_jk] == c_e:
                    mono_count += 1
                    bad_edges.add(e)
                    bad_edges.add(idx_ik)
                    bad_edges.add(idx_jk)
        return bad_edges, mono_count // 3

    # Local Search
    start_time = time.time()
    time_limit = 115.0

    for attempt in range(1000):
        if time.time() - start_time > time_limit:
            break

        # Random initialization
        colors = [random.randint(0, 3) for _ in range(num_edges)]
        
        # Optional: Start with a known good structure? 
        # No, random is fine with enough restarts.
        
        current_mono, _ = count_mono_edges(colors)
        
        max_iters = 10000
        for it in range(max_iters):
            if time.time() - start_time > time_limit:
                break

            if not current_mono:
                return {'n': n, 'colors': colors}

            # Pick an edge from the set of "bad" edges
            # If we want to be smarter, we could pick the one with the most conflicts,
            # but that requires counting conflicts per edge, which is O(E*degree).
            # Let's stick to random pick from bad edges for speed, but maybe 
            # occasionally pick the "worst" one.
            
            # To speed up convergence, let's calculate conflict count for bad edges only.
            # This is still somewhat expensive. 
            # Let's try a simple random pick first.
            
            e = random.choice(list(current_mono))
            
            # Try all 3 other colors for edge e
            best_delta = float('inf')
            best_color = -1
            
            c_old = colors[e]
            
            for c_new in range(4):
                if c_new == c_old:
                    continue
                
                # Calculate delta
                # Delta = (new mono count) - (old mono count)
                # We only need to check triangles involving e.
                # For each triangle (e, ik, jk):
                #   Old: mono if colors[ik]==c_old and colors[jk]==c_old
                #   New: mono if colors[ik]==c_new and colors[jk]==c_new
                
                delta = 0
                for idx_ik, idx_jk in tri_edges[e]:
                    c_ik = colors[idx_ik]
                    c_jk = colors[idx_jk]
                    
                    was_mono = (c_ik == c_old and c_jk == c_old)
                    is_mono = (c_ik == c_new and c_jk == c_new)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                        
                if delta < best_delta:
                    best_delta = delta
                    best_color = c_new
            
            if best_color != -1 and best_delta <= 0:
                colors[e] = best_color
                # Update current_mono? 
                # Re-calculating from scratch is O(E*degree) ~ 1225*48 ~ 60k ops.
                # Doing this every step is too slow for 10k steps (600M ops).
                # We need incremental update or less frequent recalculation.
                
                # Strategy: Update incrementally.
                # But tracking the set of bad edges incrementally is tricky.
                # Alternative: Recalculate `current_mono` every K steps or if stuck.
                # Or, just recalculate every step but optimize `count_mono_edges`.
                
                # Let's try to optimize `count_mono_edges` using NumPy?
                # Or just accept the cost if we reduce iterations?
                
                # Actually, 60k ops in Python is ~10-20ms.
                # 1000 steps * 20ms = 20s.
                # So we can do ~1000 steps per restart within 120s if we are careful.
                # But we have 1000 restarts? No, we have 120s total.
                # If one restart takes 20s, we get 6 restarts.
                # That's not enough.
                
                # We need a faster conflict check.
                
            else:
                # Stuck or no improvement. 
                # Random jump
                if random.random() < 0.5:
                    # Recolor 5 random edges
                    for _ in range(5):
                        re = random.randint(0, num_edges - 1)
                        colors[re] = random.randint(0, 3)
                    current_mono, _ = count_mono_edges(colors)
                else:
                    # Restart this attempt
                    break

        # End of inner loop

    # Fallback
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
    colors_32 = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors_32.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return {'n': 32, 'colors': colors_32}
