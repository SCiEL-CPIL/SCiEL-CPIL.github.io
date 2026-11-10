import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append((e1, e2, e3, e4, e5, e6))
        
    k4_edges_np = np.array(k4s, dtype=np.int32)  # shape (1820, 6)
    
    # Precompute: for each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx, k4 in enumerate(k4s):
        for e in k4:
            edge_to_k4s[e].append(k4_idx)
    edge_to_k4s_np = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]
    
    def count_violations_fast(colours):
        """Return number of K4s with fewer than 5 distinct colors using NumPy."""
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]  # shape (1820, 6)
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((1820, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return int(np.sum(counts < 5))

    def is_valid(colours):
        return count_violations_fast(colours) == 0

    def get_violating_k4s(colours):
        """Return list of indices of K4s that violate."""
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((1820, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return np.where(counts < 5)[0]

    def incremental_delta(colours, eidx, old_color, new_color):
        """Compute change in violation count when edge eidx changes."""
        affected = edge_to_k4s_np[eidx]
        if len(affected) == 0:
            return 0
        
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np[affected]].copy()  # shape (len(affected), 6)
        
        # Find where eidx appears in each row
        positions = np.where(k4_edges_np[affected] == eidx)
        
        # Old counts
        sorted_old = np.sort(k4_colors, axis=1)
        unique_mask_old = np.ones((len(affected), 6), dtype=bool)
        unique_mask_old[:, 1:] = (sorted_old[:, 1:] != sorted_old[:, :-1])
        old_counts = np.sum(unique_mask_old, axis=1)
        old_violating = old_counts < 5
        
        # New colors
        k4_colors[positions] = new_color
        sorted_new = np.sort(k4_colors, axis=1)
        unique_mask_new = np.ones((len(affected), 6), dtype=bool)
        unique_mask_new[:, 1:] = (sorted_new[:, 1:] != sorted_new[:, :-1])
        new_counts = np.sum(unique_mask_new, axis=1)
        new_violating = new_counts < 5
        
        return int(np.sum(new_violating) - np.sum(old_violating))

    # Start with seed
    current = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Greedy merge: try to merge color pairs to reduce count
    for target_max_colors in range(16, 12, -1):
        if time.time() > deadline:
            break
            
        if len(set(current)) <= target_max_colors:
            continue
            
        current_set = set(current)
        colors_list = sorted(current_set)
        
        if len(colors_list) <= target_max_colors:
            continue
            
        merged = False
        # Try merging pairs, prioritizing colors that appear less frequently
        color_freq = {c: current.count(c) for c in colors_list}
        colors_sorted_by_freq = sorted(colors_list, key=lambda c: color_freq[c])
        
        for i in range(len(colors_sorted_by_freq)):
            for j in range(i + 1, len(colors_sorted_by_freq)):
                if time.time() > deadline:
                    break
                
                c1 = colors_sorted_by_freq[i]
                c2 = colors_sorted_by_freq[j]
                
                new_colours = [c2 if c == c1 else c for c in current]
                
                if is_valid(new_colours):
                    current = new_colours
                    merged = True
                    break
            if merged:
                break
                
        if not merged:
            # Local search with simulated annealing
            K = target_max_colors
            # Remap colors to 0..K-1
            mapping = {}
            next_id = 0
            init_colours = []
            for c in current:
                if c not in mapping:
                    if next_id >= K:
                        # Need to drop one color
                        break
                    mapping[c] = next_id
                    next_id += 1
                init_colours.append(mapping[c])
            
            if len(init_colours) < 120:
                continue
                
            if is_valid(init_colours):
                current = init_colours
                continue
            
            best_colours = init_colours[:]
            best_violations = count_violations_fast(init_colours)
            
            # Simulated annealing
            T = 1.0
            cooling = 0.999
            for step in range(30000):
                if time.time() > deadline:
                    break
                
                if best_violations == 0:
                    break
                    
                # Pick edge from violating K4s if possible
                violating = get_violating_k4s(best_colours)
                if len(violating) > 0 and random.random() < 0.7:
                    k4_idx = violating[random.randint(0, len(violating)-1)]
                    k4_edges = k4_edges_np[k4_idx]
                    eidx = int(k4_edges[random.randint(0, 5)])
                else:
                    eidx = random.randint(0, 119)
                    
                old_color = best_colours[eidx]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                
                best_colours[eidx] = new_color
                delta = incremental_delta(best_colours, eidx, old_color, new_color)
                new_violations = best_violations + delta
                
                if new_violations <= best_violations:
                    best_violations = new_violations
                else:
                    # Accept with probability exp(-delta/T)
                    if random.random() < np.exp(-delta / T):
                        best_violations = new_violations
                    else:
                        best_colours[eidx] = old_color
                
                T *= cooling
                
            if best_violations == 0:
                current = best_colours
            else:
                # Try random restarts
                success = False
                for restart in range(30):
                    if time.time() > deadline:
                        break
                    rand_colours = [random.randint(0, K - 1) for _ in range(120)]
                    v = count_violations_fast(rand_colours)
                    
                    T = 1.0
                    cooling = 0.999
                    for step in range(15000):
                        if time.time() > deadline:
                            break
                        if v == 0:
                            success = True
                            break
                        violating = get_violating_k4s(rand_colours)
                        if len(violating) > 0 and random.random() < 0.7:
                            k4_idx = violating[random.randint(0, len(violating)-1)]
                            k4_edges = k4_edges_np[k4_idx]
                            eidx = int(k4_edges[random.randint(0, 5)])
                        else:
                            eidx = random.randint(0, 119)
                            
                        old_c = rand_colours[eidx]
                        new_c = random.randint(0, K - 1)
                        if new_c == old_c:
                            continue
                        rand_colours[eidx] = new_c
                        delta = incremental_delta(rand_colours, eidx, old_c, new_c)
                        new_v = v + delta
                        if new_v <= v:
                            v = new_v
                        else:
                            if random.random() < np.exp(-delta / T):
                                v = new_v
                            else:
                                rand_colours[eidx] = old_c
                        T *= cooling
                        
                    if success:
                        current = rand_colours
                        break
                        
                if not success:
                    break

    if not is_valid(current):
        seed = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if is_valid(seed):
            current = seed
            
    return current
