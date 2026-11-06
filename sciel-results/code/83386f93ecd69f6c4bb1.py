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
    cycles_for_edge = [[] for _ in range(E)]
    for i, cyc in enumerate(unique_cycles):
        for e in cyc:
            cycles_for_edge[e].append(i)
            
    # Precompute others arrays for vectorization
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
            
    freq = np.zeros(120, dtype=int)
    for c in colors:
        freq[c] += 1
    distinct_count = np.sum(freq > 0)
    
    bad = np.zeros(num_cycles, dtype=int)
    total_bad = 0
    for i, cyc in enumerate(unique_cycles):
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            bad[i] = 1
            total_bad += 1
            
    best_colors = colors.copy()
    best_distinct = distinct_count
    
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    # List of colors to try, prioritizing new ones
    all_colors = list(range(120))
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        e = random.randint(0, E-1)
        old_color = colors[e]
        
        # Choose new color: 50% chance pick from unused, 50% random
        if random.random() < 0.5:
            unused = [c for c in range(120) if freq[c] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, 119)
        else:
            new_color = random.randint(0, 119)
            
        if old_color == new_color:
            continue
            
        others = others_array[e]
        if others.shape[0] == 0:
            delta_bad = 0
        else:
            others_colors = colors[others]
            o0 = others_colors[:,0]
            o1 = others_colors[:,1]
            o2 = others_colors[:,2]
            
            distinct_3 = (o0 != o1) & (o1 != o2) & (o0 != o2)
            
            in_others_new = (o0 == new_color) | (o1 == new_color) | (o2 == new_color)
            new_bad = distinct_3 & ~in_others_new
            
            in_others_old = (o0 == old_color) | (o1 == old_color) | (o2 == old_color)
            old_bad = distinct_3 & ~in_others_old
            
            delta_bad = np.sum(new_bad) - np.sum(old_bad)
            
        delta_distinct = 0
        if freq[old_color] == 1:
            delta_distinct -= 1
        if freq[new_color] == 0:
            delta_distinct += 1
            
        new_total_bad = total_bad + delta_bad
        new_distinct = distinct_count + delta_distinct
        
        accept = False
        if new_total_bad == 0 and total_bad == 0:
            if new_distinct > distinct_count:
                accept = True
            elif random.random() < 0.05:
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
                elif random.random() < 0.05:
                    accept = True

        if accept:
            colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            distinct_count = new_distinct
            total_bad = new_total_bad
            
            cyc_indices = cycles_for_edge[e]
            for ci in cyc_indices:
                cyc = unique_cycles[ci]
                c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
                bad[ci] = 1 if len({c0, c1, c2, c3}) == 4 else 0
                
            if total_bad == 0:
                if distinct_count > best_distinct:
                    best_distinct = distinct_count
                    best_colors = colors.copy()
                    
        T *= cooling
        
    return best_colors.tolist()
