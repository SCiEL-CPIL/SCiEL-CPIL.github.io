"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
import random

def construct():
    n = 20
    max_colors = 20
    
    # Precompute all 4-cycles in K_n
    all_4cycles = []
    for a, b, c, d in combinations(range(n), 4):
        all_4cycles.append(((a,b), (b,c), (c,d), (d,a)))
        all_4cycles.append(((a,b), (b,d), (d,c), (c,a)))
        all_4cycles.append(((a,c), (c,b), (b,d), (d,a)))
    
    # Build edge-to-cycles incidence with positions
    edge_cycle_pos = {}
    for i, j in combinations(range(n), 2):
        key = (i, j)
        positions = []
        for idx, cycle in enumerate(all_4cycles):
            for pos, edge in enumerate(cycle):
                if (min(edge[0], edge[1]), max(edge[0], edge[1])) == key:
                    positions.append((idx, pos))
                    break
        edge_cycle_pos[key] = positions
    
    # Random edge ordering for greedy construction
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edge_count = 0
    best_result = []
    
    for trial in range(100):
        random.seed(trial)
        random.shuffle(all_edges)
        
        adj = np.full((n, n), -1, dtype=np.int32)
        cycle_colors = np.full((len(all_4cycles), 4), -1, dtype=np.int32)
        
        edge_count = 0
        
        for (u, v) in all_edges:
            key = (min(u, v), max(u, v))
            positions = edge_cycle_pos[key]
            
            colors_to_try = list(range(max_colors))
            random.shuffle(colors_to_try)
            
            added = False
            for c in colors_to_try:
                # Check validity: in every 4-cycle containing this edge,
                # no other edge in that cycle should already have color c
                valid = True
                for (cycle_idx, pos) in positions:
                    # Check if any OTHER position in this cycle already has color c
                    cc = cycle_colors[cycle_idx]
                    # Create a mask excluding current position
                    other_colors = np.concatenate([cc[:pos], cc[pos+1:]])
                    if c in other_colors:
                        valid = False
                        break
                
                if valid:
                    adj[u, v] = c
                    adj[v, u] = c
                    for (cycle_idx, pos) in positions:
                        cycle_colors[cycle_idx, pos] = c
                    edge_count += 1
                    added = True
                    break
        
        if edge_count > best_edge_count:
            best_edge_count = edge_count
            result = []
            for i in range(n):
                for j in range(i+1, n):
                    if adj[i, j] != -1:
                        result.append([i, j, int(adj[i, j])])
            best_result = result
    
    return best_result
# EVOLVE-BLOCK-END
