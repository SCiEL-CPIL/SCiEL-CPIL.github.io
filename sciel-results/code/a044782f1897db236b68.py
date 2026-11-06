"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
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
    # unique_cycles: list of tuples of 4 edge indices
    # cycles_for_edge: list of lists of tuples of 3 edge indices (others)
    
    unique_cycles = []
    seen = set()
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # 3 distinct 4-cycles on 4 vertices
        # 1: (a,b), (b,c), (c,d), (d,a)
        # 2: (a,b), (b,d), (d,c), (c,a)
        # 3: (a,c), (c,b), (b,d), (d,a)
        
        e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
        e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
        e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
        
        for cyc in (e1, e2, e3):
            key = tuple(sorted(cyc))
            if key not in seen:
                seen.add(key)
                unique_cycles.append(cyc)
                
    # Build adjacency: for each edge, which cycles does it belong to?
    # cycles_for_edge[e] = list of cycle indices (indices into unique_cycles)
    cycles_for_edge = [[] for _ in range(E)]
    for i, cyc in enumerate(unique_cycles):
        for e in cyc:
            cycles_for_edge[e].append(i)
            
    # State
    colors = np.zeros(E, dtype=int)
    freq = np.zeros(120, dtype=int)
    distinct_count = 0
    
    # Precompute badness of each cycle initially
    bad = np.zeros(len(unique_cycles), dtype=int)
    total_bad = 0
    
    # We will use a multi-restart strategy
    # Each restart: Random init, SA to fix bad, then Hill Climbing to maximize distinct.
    
    best_colors = None
    best_distinct = 0
    
    while time.time() < deadline:
        # 1. Random Initialization
        # Use a mix of random and structured init to help convergence
        # Try a few different init strategies in parallel or sequentially
        init_strategy = random.randint(0, 2)
        
        if init_strategy == 0:
            # Random
            for e in range(E):
                colors[e] = random.randint(0, 119)
        elif init_strategy == 1:
            # Gallai-like: c(i,j) = i
            for i in range(N):
                for j in range(i+1, N):
                    colors[edge_idx(i,j)] = i
        else:
            # Random but limited palette to start
            palette_size = random.randint(5, 15)
            for e in range(E):
                colors[e] = random.randint(0, palette_size - 1)
                
        # Reset state
        freq = np.zeros(120, dtype=int)
        for c in colors:
            freq[c] += 1
        distinct_count = np.sum(freq > 0)
        
        total_bad = 0
        for i, cyc in enumerate(unique_cycles):
            c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
            if len({c0, c1, c2, c3}) == 4:
                bad[i] = 1
                total_bad += 1
            else:
                bad[i] = 0
                
        # 2. Simulated Annealing to minimize total_bad
        T = 50.0
        cooling = 0.995
        min_T = 0.1
        steps_per_restart = 20000 # Adjust based on time
        
        # Time check
        if time.time() > deadline - 5:
            continue
            
        for step in range(steps_per_restart):
            if time.time() > deadline - 5:
                break
                
            if T < min_T:
                T = min_T
                
            # Pick random edge
            e = random.randint(0, E-1)
            old_color = colors[e]
            new_color = random.randint(0, 119)
            
            if old_color == new_color:
                continue
                
            # Calculate delta_bad
            delta_bad = 0
            for cyc_idx in cycles_for_edge[e]:
                cyc = unique_cycles[cyc_idx]
                others = [cyc[k] for k in range(4) if cyc[k] != e]
                c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                
                old_set = {old_color, c1, c2, c3}
                old_badness = 1 if len(old_set) == 4 else 0
                
                new_set = {new_color, c1, c2, c3}
                new_badness = 1 if len(new_set) == 4 else 0
                
                if old_badness != new_badness:
                    delta_bad += new_badness - old_badness
                    
            new_total_bad = total_bad + delta_bad
            
            # Acceptance: Minimize bad
            if delta_bad <= 0:
                accept = True
            else:
                if random.random() < np.exp(-delta_bad / T):
                    accept = True
                else:
                    accept = False
                    
            if accept:
                colors[e] = new_color
                freq[old_color] -= 1
                freq[new_color] += 1
                total_bad = new_total_bad
                
                for cyc_idx in cycles_for_edge[e]:
                    cyc = unique_cycles[cyc_idx]
                    others = [cyc[k] for k in range(4) if cyc[k] != e]
                    c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                    new_set = {new_color, c1, c2, c3}
                    bad[cyc_idx] = 1 if len(new_set) == 4 else 0
                    
            T *= cooling
            
        # 3. If valid, try to maximize distinct colors via Hill Climbing
        if total_bad == 0:
            # Hill Climbing phase
            hc_steps = 10000
            for step in range(hc_steps):
                if time.time() > deadline - 2:
                    break
                    
                e = random.randint(0, E-1)
                old_color = colors[e]
                
                # Try to find a new color that increases distinct count
                # or maintains it, without breaking validity.
                # To increase distinct, we need to pick a color not currently used, 
                # or pick a color that is used, but we lose one color elsewhere? No, 
                # changing one edge only affects its own color count.
                # If we change from old to new:
                # - If freq[old] == 1, we lose that color.
                # - If freq[new] == 0, we gain that color.
                # Net change: -1 if freq[old]==1 and freq[new]>0.
                #           +1 if freq[old]>1 and freq[new]==0.
                #           0 otherwise.
                
                # We want to maximize distinct.
                # So we prefer moves that:
                # 1. Are valid.
                # 2. Increase distinct (gain new color, don't lose old).
                # 3. Maintain distinct (shuffle).
                
                # Strategy: Pick a random unused color and see if it's valid.
                unused_colors = [c for c in range(120) if freq[c] == 0]
                if not unused_colors:
                    break # All colors used? Unlikely for 120 edges.
                
                new_color = random.choice(unused_colors)
                
                # Check validity
                delta_bad = 0
                valid = True
                for cyc_idx in cycles_for_edge[e]:
                    cyc = unique_cycles[cyc_idx]
                    others = [cyc[k] for k in range(4) if cyc[k] != e]
                    c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                    new_set = {new_color, c1, c2, c3}
                    if len(new_set) == 4:
                        valid = False
                        break
                        
                if valid:
                    # Check if we lose a color
                    if freq[old_color] > 1:
                        # We gain a new color, lose none. Distinct increases by 1.
                        # Apply
                        colors[e] = new_color
                        freq[old_color] -= 1
                        freq[new_color] += 1
                        distinct_count += 1
                    else:
                        # We gain a new color, lose old color. Distinct stays same.
                        # Accept with some probability to explore?
                        # Or reject to keep "momentum" of high distinct?
                        # Let's accept to shuffle.
                        if random.random() < 0.5:
                            colors[e] = new_color
                            freq[old_color] -= 1
                            freq[new_color] += 1
                            # distinct_count unchanged
                else:
                    # Try a used color?
                    # This might allow escape from local minima in validity? 
                    # But we are already valid. Changing to a used color might break validity.
                    # If we change to a used color, distinct might decrease or stay same.
                    # Only do this if we want to reshuffle.
                    pass
            
            # Update best
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_colors = colors.copy()
                
    if best_colors is None:
        # Fallback: Return a known valid coloring (e.g. Gallai)
        for i in range(N):
            for j in range(i+1, N):
                colors[edge_idx(i,j)] = i
        best_colors = colors.copy()
        # Calculate distinct for fallback
        freq = np.zeros(120, dtype=int)
        for c in colors:
            freq[c] += 1
        best_distinct = np.sum(freq > 0)

    return best_colors.tolist()
# EVOLVE-BLOCK-END
