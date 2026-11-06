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
            # Canonical form for deduplication (though these should be unique per quad)
            # Actually, different quads can produce the same set of edges?
            # No, a 4-cycle is defined by its 4 vertices. 
            # Wait, the edges are unique to the quad? 
            # Yes, the set of 4 edges in a simple 4-cycle uniquely identifies the 4 vertices (the cycle graph is a tree of chords? No, it's a cycle).
            # However, the same set of 4 edges might be generated from different vertex orderings?
            # No, the edges are (u,v). The set of edges is invariant.
            # So we just need to ensure we don't add the same tuple twice.
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
            
    # Initial Valid Coloring: c(i,j) = i for i < j
    # This uses colors 0..14.
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
    # bad[i] = 1 if unique_cycles[i] has 4 distinct colors, else 0
    bad = np.zeros(len(unique_cycles), dtype=int)
    total_bad = 0
    for i, cyc in enumerate(unique_cycles):
        c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
        if len({c0, c1, c2, c3}) == 4:
            bad[i] = 1
            total_bad += 1
            
    best_colors = colors.copy()
    best_distinct = distinct_count
    
    # Local Search / Simulated Annealing
    T = 100.0
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
            
        # Calculate delta_bad
        delta_bad = 0
        for cyc_idx in cycles_for_edge[e]:
            cyc = unique_cycles[cyc_idx]
            # Get colors of the other 3 edges
            others = [cyc[k] for k in range(4) if cyc[k] != e]
            c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
            
            # Old state
            old_set = {old_color, c1, c2, c3}
            old_badness = 1 if len(old_set) == 4 else 0
            
            # New state
            new_set = {new_color, c1, c2, c3}
            new_badness = 1 if len(new_set) == 4 else 0
            
            if old_badness != new_badness:
                delta_bad += new_badness - old_badness
                
        # Calculate delta_distinct
        delta_distinct = 0
        if freq[old_color] == 1:
            delta_distinct -= 1
        if freq[new_color] == 0:
            delta_distinct += 1
            
        new_total_bad = total_bad + delta_bad
        new_distinct = distinct_count + delta_distinct
        
        # Acceptance criteria
        # We want to minimize bad, maximize distinct.
        # If new state is valid (bad=0), we accept if distinct increases.
        # If new state is invalid, we accept if bad decreases.
        # We use SA to escape local minima in bad.
        
        accept = False
        
        if new_total_bad == 0 and total_bad == 0:
            # Both valid: accept if distinct increases
            if new_distinct > distinct_count:
                accept = True
            # Or with small prob to explore? No, keep it greedy for valid states.
        elif new_total_bad == 0 and total_bad > 0:
            # Found a valid state from invalid: Always accept
            accept = True
        elif new_total_bad > 0 and total_bad == 0:
            # Going from valid to invalid: Reject (unless we want to explore, but we have a best valid stored)
            # Actually, we can accept with very low prob to escape, but since we track best valid, 
            # we should probably stick to valid states if we have one.
            # However, sometimes moving to invalid is necessary to find a better valid one?
            # Given the constraint, it's risky. Let's try accepting with prob exp(-delta_bad/T)
            if random.random() < np.exp(-delta_bad / T):
                accept = True
        else:
            # Both invalid: accept if bad decreases
            if delta_bad < 0:
                accept = True
            else:
                # Accept with prob exp(-delta_bad/T)
                if delta_bad > 0:
                    if random.random() < np.exp(-delta_bad / T):
                        accept = True
                # If delta_bad == 0, accept with prob 0.5 to shuffle?
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
            for cyc_idx in cycles_for_edge[e]:
                cyc = unique_cycles[cyc_idx]
                others = [cyc[k] for k in range(4) if cyc[k] != e]
                c1, c2, c3 = colors[others[0]], colors[others[1]], colors[others[2]]
                new_set = {new_color, c1, c2, c3}
                bad[cyc_idx] = 1 if len(new_set) == 4 else 0
                
            # Update best if valid
            if total_bad == 0:
                if distinct_count > best_distinct:
                    best_distinct = distinct_count
                    best_colors = colors.copy()
                    
        T *= cooling
        
    return best_colors.tolist()
# EVOLVE-BLOCK-END
