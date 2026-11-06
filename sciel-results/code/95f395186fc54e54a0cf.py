import random
import time
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for all K4s
    def get_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)
        
    num_k4 = len(k4_edges)
    
    # Convert to numpy for vectorized operations
    k4_edges_np = np.array(k4_edges, dtype=np.int32)

    def calculate_penalty(coloring):
        """
        Returns the total penalty using vectorized operations.
        """
        colors = np.array(coloring, dtype=np.int32)
        # Get all colors for all K4s at once: shape (num_k4, 6)
        all_colors = colors[k4_edges_np]  # shape (num_k4, 6)
        
        # For each K4, count distinct colors
        # Use a trick: sort and count unique
        sorted_colors = np.sort(all_colors, axis=1)
        # Count distinct by checking where values change
        distinct_mask = np.concatenate([np.ones((num_k4, 1), dtype=bool), 
                                         sorted_colors[:, 1:] != sorted_colors[:, :-1]], axis=1)
        num_distinct = distinct_mask.sum(axis=1)
        
        # Penalty is max(0, 5 - num_distinct)
        penalty = np.maximum(0, 5 - num_distinct)
        return int(penalty.sum())

    def get_violating_k4s(coloring):
        """Return list of K4 indices that violate the constraint."""
        colors = np.array(coloring, dtype=np.int32)
        all_colors = colors[k4_edges_np]
        sorted_colors = np.sort(all_colors, axis=1)
        distinct_mask = np.concatenate([np.ones((num_k4, 1), dtype=bool), 
                                         sorted_colors[:, 1:] != sorted_colors[:, :-1]], axis=1)
        num_distinct = distinct_mask.sum(axis=1)
        violating = np.where(num_distinct < 5)[0]
        return violating.tolist()

    def is_valid(coloring):
        return calculate_penalty(coloring) == 0

    def count_colors_used(coloring):
        return len(set(coloring))

    def try_reduce_colors(coloring, max_colors):
        """Try to reduce the number of colors used in the coloring."""
        for color_to_eliminate in range(max_colors - 1, -1, -1):
            edges_with_color = [i for i, c in enumerate(coloring) if c == color_to_eliminate]
            if not edges_with_color:
                continue
            
            # Try multiple random recolorings
            for attempt in range(50):
                trial = coloring.copy()
                for e in edges_with_color:
                    trial[e] = random.randint(0, max_colors - 2)
                if is_valid(trial):
                    # Renumber colors to be compact
                    color_map = {}
                    next_color = 0
                    for c in trial:
                        if c not in color_map:
                            color_map[c] = next_color
                            next_color += 1
                    return [color_map[c] for c in trial], next_color
        return coloring, max_colors

    # Try multiple initial colorings
    initializations = []
    
    # 1. Random with 15 colors
    initializations.append([random.randint(0, 14) for _ in range(num_edges)])
    
    # 2. Modular multiplication: (i * j) mod p for prime p >= 16
    for p in [17, 19, 23]:
        init = []
        for i in range(n):
            for j in range(i + 1, n):
                init.append((i * j) % p)
        initializations.append(init)
    
    # 3. Additive: (i + j) mod p
    for p in [17, 19]:
        init = []
        for i in range(n):
            for j in range(i + 1, n):
                init.append((i + j) % p)
        initializations.append(init)
    
    # 4. Another random
    initializations.append([random.randint(0, 14) for _ in range(num_edges)])
    
    best_coloring = None
    best_penalty = float('inf')
    best_max_colors = 16
    
    for init_coloring in initializations:
        if time.time() > deadline - 30.0:
            break
        
        coloring = init_coloring.copy()
        current_max = count_colors_used(coloring)
        penalty = calculate_penalty(coloring)
        
        if penalty < best_penalty:
            best_penalty = penalty
            best_coloring = coloring.copy()
            best_max_colors = current_max
        
        if penalty == 0:
            # Try to reduce colors
            coloring, current_max = try_reduce_colors(coloring, current_max)
            if current_max < best_max_colors:
                best_max_colors = current_max
                best_coloring = coloring.copy()
                best_penalty = 0
            continue
        
        # Simulated Annealing
        T = 50.0
        cooling_rate = 0.9997
        min_T = 0.1
        local_steps = 0
        max_local_steps = 50000
        no_improve = 0
        best_local_penalty = penalty
        best_local_coloring = coloring.copy()
        
        while T > min_T and local_steps < max_local_steps and time.time() < deadline - 20.0:
            # Targeted move: pick an edge from a violating K4
            if best_local_penalty > 0 and random.random() < 0.5:
                violating = get_violating_k4s(best_local_coloring)
                if violating:
                    k4_idx = random.choice(violating)
                    edge_indices = k4_edges[k4_idx]
                    e_idx = random.choice(edge_indices)
                else:
                    e_idx = random.randint(0, num_edges - 1)
            else:
                e_idx = random.randint(0, num_edges - 1)
            
            old_color = coloring[e_idx]
            new_color = random.randint(0, best_max_colors - 1)
            
            if old_color == new_color:
                local_steps += 1
                continue
                
            coloring[e_idx] = new_color
            new_penalty = calculate_penalty(coloring)
            local_steps += 1
            
            delta = new_penalty - best_local_penalty
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                if new_penalty < best_local_penalty:
                    best_local_penalty = new_penalty
                    best_local_coloring = coloring.copy()
                    no_improve = 0
                    # If we found valid, try reducing
                    if best_local_penalty == 0:
                        coloring, current_max = try_reduce_colors(coloring, best_max_colors)
                        if current_max < best_max_colors:
                            best_max_colors = current_max
                            best_coloring = coloring.copy()
                            best_penalty = 0
                        T = 5.0
                else:
                    no_improve += 1
            else:
                coloring[e_idx] = old_color
                no_improve += 1
            
            T *= cooling_rate
            
            if best_local_penalty == 0 and best_max_colors <= 13:
                break
        
        if best_local_penalty < best_penalty:
            best_penalty = best_local_penalty
            best_coloring = best_local_coloring.copy()
            best_max_colors = count_colors_used(best_local_coloring)
    
    # Final polish: if we have a valid solution, try harder to reduce colors
    if best_penalty == 0:
        for _ in range(10):
            if time.time() > deadline - 10.0:
                break
            coloring, current_max = try_reduce_colors(best_coloring, best_max_colors)
            if current_max < best_max_colors:
                best_max_colors = current_max
                best_coloring = coloring
    
    # If we found a valid coloring, return it
    if best_penalty == 0:
        # Ensure colors are in 0..119
        return best_coloring
    
    # Fallback: Try more aggressive random restarts
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        coloring = [random.randint(0, 14) for _ in range(num_edges)]
        T = 30.0
        for _ in range(3000):
            if time.time() > deadline - 5.0:
                break
            e_idx = random.randint(0, num_edges - 1)
            old_color = coloring[e_idx]
            new_color = random.randint(0, 14)
            if old_color == new_color: continue
            coloring[e_idx] = new_color
            new_penalty = calculate_penalty(coloring)
            delta = new_penalty - best_penalty
            if delta < 0 or random.random() < pow(2.718281828, -delta / T):
                best_penalty = new_penalty
                best_coloring = coloring.copy()
            else:
                coloring[e_idx] = old_color
            T *= 0.999
        if best_penalty == 0:
            return best_coloring
            
    # If still invalid, return a hardcoded valid solution
    # (i+j)%17 uses 17 colors and is valid
    return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
