import time
import random
import numpy as np

N = 16
N_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def construct():
    deadline = time.time() + 55
    n_edges = N_EDGES
    
    edges_list = []
    for i in range(N):
        for j in range(i + 1, N):
            edges_list.append((i, j))
    
    # Build 4-cycle data
    edge_cycles = [[] for _ in range(n_edges)]
    for idx, (u, v) in enumerate(edges_list):
        others = [x for x in range(N) if x != u and x != v]
        for w_idx in range(len(others)):
            for x_idx in range(w_idx + 1, len(others)):
                w = others[w_idx]
                x = others[x_idx]
                e1 = _edge_idx(u, w)
                e2 = _edge_idx(w, x)
                e3 = _edge_idx(x, v)
                edge_cycles[idx].append((e1, e2, e3))
                e4 = _edge_idx(u, x)
                e5 = _edge_idx(x, w)
                e6 = _edge_idx(w, v)
                edge_cycles[idx].append((e4, e5, e6))
    
    cycle_data = []
    for idx in range(n_edges):
        cycles = edge_cycles[idx]
        if cycles:
            arr = np.array(cycles, dtype=np.int32)
        else:
            arr = np.empty((0, 3), dtype=np.int32)
        cycle_data.append(arr)

    def build_gallai(vertices, colors, color_counter_holder):
        if len(vertices) <= 1:
            return
        mid = len(vertices) // 2
        v1 = vertices[:mid]
        v2 = vertices[mid:]
        c = color_counter_holder[0]
        color_counter_holder[0] += 1
        for u in v1:
            for v in v2:
                colors[_edge_idx(u, v)] = c
        build_gallai(v1, colors, color_counter_holder)
        build_gallai(v2, colors, color_counter_holder)

    def is_valid_change(edge_idx, new_color, cur_colors):
        cycles = cycle_data[edge_idx]
        if len(cycles) == 0:
            return True
        c1 = cur_colors[cycles[:, 0]]
        c2 = cur_colors[cycles[:, 1]]
        c3 = cur_colors[cycles[:, 2]]
        all_distinct = (c1 != c2) & (c1 != c3) & (c2 != c3)
        new_distinct = (new_color != c1) & (new_color != c2) & (new_color != c3)
        invalid = all_distinct & new_distinct
        return not np.any(invalid)

    def run_search(init_colors, time_limit):
        colors = init_colors.copy()
        color_counts = np.zeros(120, dtype=np.int32)
        for c in colors:
            color_counts[c] += 1
        
        used_colors = set(colors.tolist())
        unused_colors = set(range(120)) - used_colors
        
        best_colors = colors.copy()
        best_distinct = len(used_colors)
        
        end_time = time.time() + time_limit
        rng = random.Random()
        
        while time.time() < end_time:
            e = rng.randint(0, n_edges - 1)
            old_c = int(colors[e])
            
            if len(unused_colors) > 0:
                if rng.random() < 0.8:
                    c = rng.choice(list(unused_colors))
                else:
                    c = rng.randint(0, 119)
            else:
                c = rng.randint(0, 119)
            
            if c == old_c:
                continue
            
            if is_valid_change(e, c, colors):
                colors[e] = c
                color_counts[old_c] -= 1
                color_counts[c] += 1
                
                if color_counts[old_c] == 0:
                    unused_colors.add(old_c)
                if c in unused_colors:
                    unused_colors.discard(c)
                
                current_distinct = len(used_colors)
                # Optimization: update used_colors set only if necessary?
                # set update is O(1) amortized.
                # But len(set) is O(N).
                # We can track count.
                # Actually, we can just compute len(set(colors.tolist())) which is slow.
                # Let's maintain a set `used_colors` manually.
                
                # The code above doesn't update used_colors set!
                # It updates color_counts but not the set.
                # `current_distinct = len(set(colors.tolist()))` is very slow (O(N^2) effectively if done every step? No, O(N) to list, O(N) to set).
                # N=120. 120 ops per step.
                # 100,000 steps/s -> 12M ops/s. This is significant.
                
                # Let's fix the tracking.
                pass

        return best_colors, best_distinct

    # We need to fix the tracking in run_search
    # Let's rewrite run_search properly.

    def run_search_v2(init_colors, time_limit):
        colors = init_colors.copy()
        color_counts = np.zeros(120, dtype=np.int32)
        for c in colors:
            color_counts[c] += 1
        
        used_colors = set(colors.tolist())
        unused_colors = set(range(120)) - used_colors
        
        best_colors = colors.copy()
        best_distinct = len(used_colors)
        
        end_time = time.time() + time_limit
        rng = random.Random()
        
        while time.time() < end_time:
            e = rng.randint(0, n_edges - 1)
            old_c = int(colors[e])
            
            if len(unused_colors) > 0:
                if rng.random() < 0.8:
                    c = rng.choice(list(unused_colors))
                else:
                    c = rng.randint(0, 119)
            else:
                c = rng.randint(0, 119)
            
            if c == old_c:
                continue
            
            if is_valid_change(e, c, colors):
                # Update counts
                color_counts[old_c] -= 1
                color_counts[c] += 1
                
                # Update sets
                if color_counts[old_c] == 0:
                    used_colors.discard(old_c)
                    unused_colors.add(old_c)
                
                if color_counts[c] == 1: # Was 0 before increment?
                     # We incremented to 1, so it was 0.
                    used_colors.add(c)
                    unused_colors.discard(c)
                
                colors[e] = c
                
                current_distinct = len(used_colors)
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors.copy()

        return best_colors, best_distinct

    best_global = None
    best_global_distinct = 0
    
    # Try a few different initializations
    num_restarts = 5
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline - 1: break
        
        # Generate a random Gallai coloring
        vertices = list(range(N))
        random.shuffle(vertices)
        init_colors = np.zeros(n_edges, dtype=np.int32)
        color_counter = [0]
        build_gallai(vertices, init_colors, color_counter)
        
        bc, bd = run_search_v2(init_colors, time_per_restart)
        
        if bd > best_global_distinct:
            best_global_distinct = bd
            best_global = bc
    
    if best_global is None:
        # Fallback
        vertices = list(range(N))
        init_colors = np.zeros(n_edges, dtype=np.int32)
        color_counter = [0]
        build_gallai(vertices, init_colors, color_counter)
        return init_colors.tolist()
        
    return best_global.tolist()
