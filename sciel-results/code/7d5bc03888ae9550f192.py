import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
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
    
    def count_violations(colours):
        """Return number of K4s with fewer than 5 distinct colors using NumPy."""
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]  # shape (1820, 6)
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((1820, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return int(np.sum(counts < 5))
    
    def is_valid(colours):
        return count_violations(colours) == 0
    
    # Try multiple starting points to find a good valid coloring
    best_valid = None
    best_num_colors = 999
    
    # Starting point 1: (i+j) % 17
    init1 = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    if is_valid(init1):
        best_valid = init1
        best_num_colors = len(set(init1))
    
    # Starting point 2: (i+j) % 15
    init2 = [(i + j) % 15 for i in range(n) for j in range(i + 1, n)]
    if is_valid(init2):
        nc = len(set(init2))
        if nc < best_num_colors:
            best_valid = init2
            best_num_colors = nc
    
    # Starting point 3: random
    for _ in range(20):
        if time.time() > deadline:
            break
        rand_init = [random.randint(0, 14) for _ in range(120)]
        if is_valid(rand_init):
            nc = len(set(rand_init))
            if nc < best_num_colors:
                best_valid = rand_init
                best_num_colors = nc
    
    if best_valid is None:
        # Fallback: use the seed
        return init1
    
    current = best_valid[:]
    
    # Try to reduce colors by merging
    for target_K in range(max(13, best_num_colors - 1), 12, -1):
        if time.time() > deadline - 5:
            break
        
        if best_num_colors <= target_K:
            continue
        
        # Try to merge colors in current to reduce to target_K
        current_set = set(current)
        if len(current_set) <= target_K:
            continue
        
        colors_list = sorted(current_set)
        
        # Greedy merging: try to merge pairs of colors
        merged = True
        while merged and len(set(current)) > target_K:
            if time.time() > deadline - 5:
                break
            
            merged = False
            current_set = set(current)
            colors_list = sorted(current_set)
            
            for i in range(len(colors_list)):
                for j in range(i + 1, len(colors_list)):
                    if time.time() > deadline - 5:
                        break
                    
                    c1 = colors_list[i]
                    c2 = colors_list[j]
                    
                    new_colours = [c2 if c == c1 else c for c in current]
                    
                    if is_valid(new_colours):
                        current = new_colours
                        best_num_colors = len(set(current))
                        merged = True
                        break
                if merged:
                    break
        
        # If we couldn't merge down to target_K, try local search
        if len(set(current)) > target_K:
            K = target_K
            mapping = {}
            next_id = 0
            init_colours = []
            for c in current:
                if c not in mapping:
                    if next_id >= K:
                        break
                    mapping[c] = next_id
                    next_id += 1
                init_colours.append(mapping[c])
            
            if len(init_colours) < 120:
                continue
            
            # Local search to fix violations
            best_colours = init_colours[:]
            best_violations = count_violations(init_colours)
            
            # Simulated annealing style local search
            no_improvement = 0
            for step in range(3000):
                if time.time() > deadline - 5:
                    break
                
                if best_violations == 0:
                    current = best_colours
                    best_num_colors = K
                    break
                
                # Pick a random edge
                eidx = random.randint(0, 119)
                old_color = best_colours[eidx]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                
                best_colours[eidx] = new_color
                new_violations = count_violations(best_colours)
                
                # Accept if better or equal, or with small probability if worse
                if new_violations <= best_violations:
                    best_violations = new_violations
                    no_improvement = 0
                else:
                    # Accept worse with small probability to escape local minima
                    if random.random() < 0.1:
                        best_violations = new_violations
                        no_improvement += 1
                    else:
                        best_colours[eidx] = old_color
                        no_improvement += 1
                
                if no_improvement > 200:
                    break
            
            if best_violations == 0:
                current = best_colours
                best_num_colors = K
    
    # Final check
    if not is_valid(current):
        return best_valid
    
    return current
