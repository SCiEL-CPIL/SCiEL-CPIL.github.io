import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs: (i0, i1, i2, i3) 0-indexed
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Build per-position list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    # Precompute "others" for each position to speed up flip_delta
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p]
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)
    
    def count_mono(color):
        cols = color[aps_arr]
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    def flip_delta(color, p):
        others = pos_others[p]
        if len(others) == 0:
            return 0
        other_colors = color[others]
        cp = color[p]
        new_cp = 1 - cp
        all_same_now = np.all(other_colors == cp, axis=1)
        all_same_new = np.all(other_colors == new_cp, axis=1)
        delta = np.sum((~all_same_now) & all_same_new) - np.sum(all_same_now & (~all_same_new))
        return int(delta)
    
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = bin(i).count('1') % 2
        return color
    
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color
    
    # Improved local search with multiple sweep orders
    def local_search(color, max_iter=5000):
        color = color.copy()
        current_score = count_mono(color)
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            best_delta = 0
            best_pos = -1
            for p in range(n):
                d = flip_delta(color, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            if best_pos >= 0:
                color[best_pos] = 1 - color[best_pos]
                current_score += best_delta
                improved = True
        return color, current_score
    
    # Simulated annealing with adaptive cooling
    def simulated_annealing(color, max_iter=1000, start_temp=5.0, end_temp=0.01):
        color = color.copy()
        current_score = count_mono(color)
        best_score = current_score
        best_color = color.copy()
        
        for i in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Geometric cooling
            ratio = i / max_iter
            temp = start_temp * (end_temp / start_temp) ** ratio
            
            p = np.random.randint(0, n)
            delta = flip_delta(color, p)
            
            # Accept if improvement or with probability exp(-delta/temp)
            if delta < 0 or (temp > 0 and np.random.random() < np.exp(-delta / temp)):
                color[p] = 1 - color[p]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
        
        return best_color, best_score
    
    # Iterated local search with simulated annealing perturbation
    def iterated_local_search(color, max_iterations=100):
        color = color.copy()
        best_color = color.copy()
        best_score = count_mono(color)
        stuck_count = 0
        
        for it in range(max_iterations):
            if time.time() - start_time > deadline - 5:
                break
            
            # Local search
            color, score = local_search(color, max_iter=2000)
            
            if score < best_score:
                best_score = score
                best_color = color.copy()
                stuck_count = 0
            else:
                stuck_count += 1
            
            # Adaptive perturbation based on how stuck we are
            if stuck_count >= 5:
                # Use SA for larger escape
                num_flips = np.random.randint(8, 20)
            elif stuck_count >= 2:
                num_flips = np.random.randint(4, 12)
            else:
                num_flips = np.random.randint(2, 8)
            
            flip_positions = np.random.choice(n, size=min(num_flips, n), replace=False)
            color[flip_positions] = 1 - color[flip_positions]
        
        return best_color, best_score
    
    # Generate diverse starting points
    startings = [thue_morse(n)]
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(10):
        startings.append(random_coloring())
    
    # Thue-Morse with random perturbations
    for _ in range(8):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=10, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    # Phase 1: Quick local search on all startings
    for start_color in startings:
        if time.time() - start_time > deadline - 30:
            break
        color, score = local_search(start_color, max_iter=3000)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # Phase 2: Iterated local search on best candidates
    if best_overall_color is not None and time.time() - start_time < deadline - 20:
        # Run ILS on the best found
        color, score = iterated_local_search(best_overall_color, max_iterations=150)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # Phase 3: SA polish on the best
    if best_overall_color is not None and time.time() - start_time < deadline - 10:
        color, score = simulated_annealing(best_overall_color, max_iter=2000, start_temp=3.0, end_temp=0.005)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
        
        # Final local search polish
        if time.time() - start_time < deadline - 3:
            color, score = local_search(best_overall_color, max_iter=5000)
            if score < best_overall_score:
                best_overall_score = score
                best_overall_color = color.copy()
    
    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
