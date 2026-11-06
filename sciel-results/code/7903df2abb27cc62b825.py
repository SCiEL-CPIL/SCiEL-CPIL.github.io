import random
import time
import itertools
import numpy as np

N = 16
E = N*(N-1)//2  # 120 edges

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i*(2*N-1-i)//2 + j - i - 1

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Precompute cycles
    unique_cycles = []
    seen = set()
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        
        for cyc in (e1, e2, e3):
            key = tuple(sorted(cyc))
            if key not in seen:
                seen.add(key)
                unique_cycles.append(cyc)
                
    num_cycles = len(unique_cycles)
    
    # Build adjacency: for each edge, which cycles does it belong to?
    # cycles_for_edge[e] = list of cycle indices
    cycles_for_edge = [[] for _ in range(E)]
    # others_for_edge[e][cyc_idx] = list of 3 other edge indices
    others_for_edge = [None] * E
    
    for i, cyc in enumerate(unique_cycles):
        for e in cyc:
            cycles_for_edge[e].append(i)
            
    # Precompute others arrays for vectorization
    # others_array[e] is a 2D array of shape (num_cycles_for_e, 3)
    others_array = [None] * E
    for e in range(E):
        cyc_indices = cycles_for_edge[e]
        if not cyc_indices:
            others_array[e] = np.empty((0,3), dtype=int)
            continue
        others = []
        for ci in cyc_indices:
            cyc = unique_cycles[ci]
            others.append([cyc[k] for k in range(4) if cyc[k] != e])
        others_array[e] = np.array(others, dtype=int)
        
    # Initial Valid Coloring: c(i,j) = i for i < j
    colors = np.zeros(E, dtype=int)
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i,j)] = i
            
    # State
    freq = np.zeros(120, dtype=int)
    for c in colors:
        freq[c] += 1
    distinct_count = np.sum(freq > 0)
    
    # Precompute badness of each cycle initially
    bad = np.zeros(num_cycles, dtype=int)
    total_bad = 0
    for i, cyc in enumerate(unique_cycles):
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            bad[i] = 1
            total_bad += 1
            
    best_colors = colors.copy()
    best_distinct = distinct_count
    
    # Local Search / Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        e = random.randint(0, E-1)
        old_color = colors[e]
        new_color = random.randint(0, 119)
        
        if old_color == new_color:
            continue
            
        # Calculate delta_bad using vectorized check
        # Get others colors for all cycles containing e
        others = others_array[e]
        if others.shape[0] == 0:
            delta_bad = 0
        else:
            others_colors = colors[others] # Shape (num_cycs, 3)
            # Check if len({c_new} U {c_others}) == 4
            # This is true if:
            # 1. The 3 others are distinct (len == 3)
            # 2. c_new is not in the others
            
            # Count distinct in others
            # We can check if any pair is equal
            # Or just use a set logic per row?
            # Since 3 elements, we can check:
            # distinct_3 = (others_colors[:,0] != others_colors[:,1]) & 
            #              (others_colors[:,1] != others_colors[:,2]) & 
            #              (others_colors[:,0] != others_colors[:,2])
            
            o0 = others_colors[:,0]
            o1 = others_colors[:,1]
            o2 = others_colors[:,2]
            
            distinct_3 = (o0 != o1) & (o1 != o2) & (o0 != o2)
            
            # c_new in others?
            in_others = (o0 == new_color) | (o1 == new_color) | (o2 == new_color)
            
            # New badness
            new_bad = distinct_3 & ~in_others
            
            # Old badness
            # We can compute old_bad similarly, or use the precomputed bad array
            # But bad array is global. We need badness of these specific cycles.
            # Let's compute old_bad for these cycles too.
            # Old color is old_color.
            in_others_old = (o0 == old_color) | (o1 == old_color) | (o2 == old_color)
            old_bad = distinct_3 & ~in_others_old
            
            delta_bad = np.sum(new_bad) - np.sum(old_bad)
            
        # Calculate delta_distinct
        delta_distinct = 0
        if freq[old_color] == 1:
            delta_distinct -= 1
        if freq[new_color] == 0:
            delta_distinct += 1
            
        new_total_bad = total_bad + delta_bad
        new_distinct = distinct_count + delta_distinct
        
        # Acceptance criteria
        accept = False
        
        if new_total_bad == 0 and total_bad == 0:
            if new_distinct > distinct_count:
                accept = True
            elif random.random() < 0.1:
                accept = True
        elif new_total_bad == 0 and total_bad > 0:
            accept = True
        elif new_total_bad > 0 and total_bad == 0:
            if random.random() < np.exp(-delta_bad / T):
                accept = True
        else:
            if delta_bad < 0:
                accept = True
            else:
                if delta_bad > 0:
                    if random.random() < np.exp(-delta_bad / T):
                        accept = True
                elif random.random() < 0.1:
                    accept = True

        if accept:
            # Apply move
            colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            distinct_count = new_distinct
            total_bad = new_total_bad
            
            # Update bad array for affected cycles
            # We can just recompute bad for these cycles
            cyc_indices = cycles_for_edge[e]
            for ci in cyc_indices:
                cyc = unique_cycles[ci]
                c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                bad[ci] = 1 if len({c0, c1, c2, c3}) == 4 else 0
                
            # Update best if valid
            if total_bad == 0:
                if distinct_count > best_distinct:
                    best_distinct = distinct_count
                    best_colors = colors.copy()
                    
        T *= cooling
        
    return best_colors.tolist()
