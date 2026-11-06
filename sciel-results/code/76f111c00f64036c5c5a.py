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
    
    # Precompute all 4-cycles (rainbow constraints)
    # A 4-cycle is a set of 4 edges forming a cycle on 4 vertices
    # For vertices a,b,c,d, the three possible 4-cycles are:
    # (a,b,c,d,a), (a,b,d,c,a), (a,c,b,d,a)
    # We need: no cycle has 4 distinct colors
    
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
    
    # For each edge, which cycles does it belong to?
    cycles_for_edge = [[] for _ in range(E)]
    for i, cyc in enumerate(unique_cycles):
        for e in cyc:
            cycles_for_edge[e].append(i)
    
    # For each cycle, which edges does it contain?
    edges_in_cycle = [list(cyc) for cyc in unique_cycles]
    
    # Strategy: Greedy assignment with smart ordering
    # 1. Order edges to assign (by degree in cycle graph - edges in more cycles first)
    # 2. For each edge, try colors in order: prefer unused colors, then used colors
    # 3. Check validity incrementally
    
    # Order edges: edges that appear in more cycles are more constrained, assign them first
    edge_order = sorted(range(E), key=lambda e: -len(cycles_for_edge[e]))
    
    best_colors = None
    best_distinct = 0
    
    # Try multiple random restarts with different color orderings
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Random shuffle of color preference order
        color_order = list(range(120))
        random.shuffle(color_order)
        
        colors = np.full(E, -1, dtype=int)
        freq = np.zeros(120, dtype=int)
        
        # Track which cycles are "active" (have some edges assigned)
        # and their current color sets
        cycle_colors = [set() for _ in range(num_cycles)]
        
        valid = True
        distinct_count = 0
        
        for e in edge_order:
            if time.time() > deadline - 3.0:
                valid = False
                break
                
            # Get the cycles this edge belongs to
            my_cycles = cycles_for_edge[e]
            
            # Try each color in our preferred order
            assigned = False
            for c in color_order:
                # Check if assigning color c to edge e creates a rainbow cycle
                conflict = False
                for ci in my_cycles:
                    # Get the other 3 edges in this cycle
                    cyc_edges = edges_in_cycle[ci]
                    other_edges = [x for x in cyc_edges if x != e]
                    
                    # Get their colors (if assigned)
                    other_colors = []
                    for oe in other_edges:
                        if colors[oe] >= 0:
                            other_colors.append(colors[oe])
                    
                    # If all 3 others are assigned and distinct, and c is different from all,
                    # then this would create a rainbow cycle
                    if len(other_colors) == 3:
                        if len(set(other_colors)) == 3 and c not in other_colors:
                            conflict = True
                            break
                    
                    # Also check: if 2 others are assigned and distinct, and the 3rd is unassigned,
                    # we need to be careful but for now just check the full case
                    
                if not conflict:
                    # Assign this color
                    colors[e] = c
                    freq[c] += 1
                    if freq[c] == 1:
                        distinct_count += 1
                    
                    # Update cycle color sets
                    for ci in my_cycles:
                        cycle_colors[ci].add(c)
                    
                    assigned = True
                    break
            
            if not assigned:
                # No valid color found - this restart failed
                valid = False
                break
        
        if valid and distinct_count > best_distinct:
            best_distinct = distinct_count
            best_colors = colors.copy()
    
    if best_colors is None:
        # Fallback: simple valid coloring
        best_colors = np.zeros(E, dtype=int)
        for i in range(N):
            for j in range(i+1, N):
                best_colors[edge_idx(i,j)] = i
        best_distinct = N
    
    return best_colors.tolist()
