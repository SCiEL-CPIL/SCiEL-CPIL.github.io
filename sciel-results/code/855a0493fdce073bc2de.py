"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

N = 64

# EVOLVE-BLOCK-START
def construct():
    # Generate grid points
    points = [(x, y) for x in range(N) for y in range(N)]
    num_points = len(points)
    
    # Precompute adjacency bitsets
    # adj[i] is an integer where bit j is 1 if i and j conflict
    adj = [0] * num_points
    
    # Helper to add edge
    def add_edge(u, v):
        if u == v: return
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Build conflict graph
    # Two points u, v conflict if there exists w in grid such that dist(u,w) == dist(v,w)
    # This is equivalent to w being on the perpendicular bisector of uv.
    # Optimization: Only check if u, v have same parity of (x+y).
    # If different parity, bisector has no integer points.
    
    # To speed up, we can group points by parity.
    # But we need to add edges.
    # Instead of checking all pairs, we can iterate w and add cliques?
    # No, that's what the old code did.
    # Let's just check pairs. 4096^2 / 2 = 8M pairs.
    # For each pair, check parity. If same, check if bisector has point in grid.
    # This might be slow in Python if done naively.
    
    # Faster: Iterate w. For each w, find all pairs (u,v) at same dist from w.
    # This creates cliques.
    # We can use the old neighbor logic but optimize it.
    
    neighbors = [{} for _ in range(num_points)]
    for i in range(num_points):
        x1, y1 = points[i]
        for j in range(num_points):
            if i == j: continue
            x2, y2 = points[j]
            dx = x1 - x2
            dy = y1 - y2
            d2 = dx*dx + dy*dy
            if d2 not in neighbors[i]:
                neighbors[i][d2] = []
            neighbors[i][d2].append(j)
            
    # Add edges from Type 1 (w is apex)
    for w in range(num_points):
        for d2, lst in neighbors[w].items():
            k = len(lst)
            if k < 2: continue
            for a in range(k):
                u = lst[a]
                for b in range(a+1, k):
                    v = lst[b]
                    add_edge(u, v)

    # Solve MIS using B&B
    best_size = 0
    best_set = []
    
    def bit_count(x):
        return bin(x).count('1')
        
    def get_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb
            
    def search(candidates_mask, current_set):
        nonlocal best_size, best_set
        
        # Pruning
        if bit_count(candidates_mask) + len(current_set) <= best_size:
            return
            
        if candidates_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
            
        # Heuristic: Pick vertex with max degree in candidates
        best_v = -1
        max_deg = -1
        for v in get_bits(candidates_mask):
            deg = bit_count(adj[v] & candidates_mask)
            if deg > max_deg:
                max_deg = deg
                best_v = v
                
        if best_v == -1:
            return
            
        v_bit = 1 << best_v
        
        # Branch 1: Include v
        new_candidates = candidates_mask & ~(adj[best_v] | v_bit)
        current_set.append(best_v)
        search(new_candidates, current_set)
        current_set.pop()
        
        # Branch 2: Exclude v
        new_candidates = candidates_mask & ~v_bit
        search(new_candidates, current_set)
        
    # Initial candidates: all points
    initial_mask = (1 << num_points) - 1
    search(initial_mask, [])
    
    return [points[i] for i in best_set]

# EVOLVE-BLOCK-END
