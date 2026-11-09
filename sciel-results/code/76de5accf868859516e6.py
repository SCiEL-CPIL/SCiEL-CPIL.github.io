"""Dense 4-cycle rainbow edge-colouring on 20 vertices."""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 20
    num_colors = 20
    
    # Bipartite split: vertices 0..9 on left, 10..19 on right
    u_nodes = list(range(10))
    v_nodes = list(range(10, 20))
    
    # Precompute all 4-cycles in the 10x10 bipartite graph
    # A 4-cycle is determined by (i1,i2,j1,j2) with i1<i2, j1<j2
    # Edges: (i1,j1), (i2,j1), (i2,j2), (i1,j2)
    cycle_list = []
    for i1 in range(10):
        for i2 in range(i1 + 1, 10):
            for j1 in range(10):
                for j2 in range(j1 + 1, 10):
                    cycle_list.append(((i1, j1), (i2, j1), (i2, j2), (i1, j2)))
    
    num_cycles = len(cycle_list)
    
    # For each cell (i,j), store which cycles it belongs to
    cell_to_cycles = defaultdict(list)
    for idx, cyc in enumerate(cycle_list):
        for (r, c) in cyc:
            cell_to_cycles[(r, c)].append(idx)
    
    # Precompute for each cycle its 4 edge positions as flat indices
    # Flat index: i*10 + j
    cycle_edge_flat = np.zeros((num_cycles, 4), dtype=np.int32)
    for idx, cyc in enumerate(cycle_list):
        for k, (r, c) in enumerate(cyc):
            cycle_edge_flat[idx, k] = r * 10 + c
    
    # For each cell, the list of (cycle_idx, position_in_cycle)
    cell_cycle_pos = defaultdict(list)
    for idx in range(num_cycles):
        for k in range(4):
            flat = cycle_edge_flat[idx, k]
            r, c = divmod(flat, 10)
            cell_cycle_pos[(r, c)].append((idx, k))
    
    def count_bad_cycles(M_flat):
        """M_flat: 100-element array, -1 = absent, else color 0..19"""
        # For each cycle, get the 4 colors
        cols = M_flat[cycle_edge_flat]  # shape (num_cycles, 4)
        all_present = (cols != -1).all(axis=1)
        
        c0 = cols[:, 0]
        c1 = cols[:, 1]
        c2 = cols[:, 2]
        c3 = cols[:, 3]
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & \
                   (c1 != c2) & (c1 != c3) & (c2 != c3)
        
        bad_mask = all_present & ~distinct
        return int(np.sum(bad_mask)), bad_mask
    
    def get_edge_count(M_flat):
        return int(np.sum(M_flat != -1))
    
    random.seed(42)
    
    best_M = None
    best_edge_count = 0
    
    # === PHASE 1: Greedy with forward checking ===
    # Try multiple random orderings; for each cell, pick a color that doesn't create bad cycles
    num_greedy_attempts = 40
    for attempt in range(num_greedy_attempts):
        if time.time() > deadline - 30.0:
            break
        
        M = np.full(100, -1, dtype=np.int32)
        
        # Random cell ordering
        cells = list(range(100))
        random.shuffle(cells)
        
        # For forward checking: track for each cycle how many edges are colored and their colors
        # We'll just recompute bad count efficiently per addition
        
        for flat_idx in cells:
            if time.time() > deadline - 30.0:
                break
            i, j = divmod(flat_idx, 10)
            
            # Try colors in random order
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            found = False
            for color in colors:
                M[flat_idx] = color
                # Quick check: only need to check cycles involving this cell
                bad = 0
                for (cyc_idx, pos) in cell_cycle_pos[(i, j)]:
                    # Check if this cycle is now bad
                    c0 = M[cycle_edge_flat[cyc_idx, 0]]
                    c1 = M[cycle_edge_flat[cyc_idx, 1]]
                    c2 = M[cycle_edge_flat[cyc_idx, 2]]
                    c3 = M[cycle_edge_flat[cyc_idx, 3]]
                    if -1 in (c0, c1, c2, c3):
                        continue
                    if len(set([c0, c1, c2, c3])) < 4:
                        bad += 1
                        break
                if bad == 0:
                    found = True
                    break
                else:
                    M[flat_idx] = -1
            
            if not found:
                M[flat_idx] = -1
        
        edge_count = get_edge_count(M)
        bad_count, _ = count_bad_cycles(M)
        
        if bad_count == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_M = M.copy()
    
    # === PHASE 2: Simulated annealing on full 10x10 matrix ===
    # Start from a random full coloring, minimize bad cycles, then try to remove edges
    if time.time() < deadline - 20.0:
        best_sa_edges = best_edge_count
        best_sa_M = best_M.copy() if best_M is not None else None
        
        num_sa_restarts = 8
        for sa_restart in range(num_sa_restarts):
            if time.time() > deadline - 15.0:
                break
            
            M = np.random.randint(0, num_colors, size=100, dtype=np.int32)
            current_bad, _ = count_bad_cycles(M)
            
            T = 5.0
            T_min = 0.1
            cooling = 0.995
            local_best = M.copy()
            local_best_bad = current_bad
            local_best_edges = get_edge_count(M)
            
            max_steps = 5000
            for step in range(max_steps):
                if time.time() > deadline - 15.0:
                    break
                if current_bad == 0:
                    break
                
                # Pick a random cell to recolor
                flat_idx = random.randint(0, 99)
                old_color = M[flat_idx]
                new_color = random.randint(0, num_colors - 1)
                if new_color == old_color:
                    continue
                M[flat_idx] = new_color
                
                new_bad, _ = count_bad_cycles(M)
                delta = new_bad - current_bad
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    current_bad = new_bad
                    if current_bad < local_best_bad:
                        local_best_bad = current_bad
                        local_best = M.copy()
                        local_best_edges = get_edge_count(M)
                else:
                    M[flat_idx] = old_color
                
                T *= cooling
            
            # Now try to remove edges to reduce bad cycles further
            # If local_best has 0 bad cycles, use it directly
            if local_best_bad == 0:
                edges = get_edge_count(local_best)
                if edges > best_sa_edges:
                    best_sa_edges = edges
                    best_sa_M = local_best.copy()
            else:
                # Try removing edges from local_best to fix remaining bad cycles
                M2 = local_best.copy()
                bad_count2, bad_mask2 = count_bad_cycles(M2)
                
                # Iteratively remove edges from bad cycles
                for _ in range(200):
                    if time.time() > deadline - 15.0:
                        break
                    if bad_count2 == 0:
                        break
                    
                    bad_indices = np.where(bad_mask2)[0]
                    if len(bad_indices) == 0:
                        break
                    target = bad_indices[random.randint(0, len(bad_indices) - 1)]
                    
                    # Pick a random edge in this cycle and remove it
                    pos = random.randint(0, 3)
                    flat_idx = cycle_edge_flat[target, pos]
                    M2[flat_idx] = -1
                    bad_count2, bad_mask2 = count_bad_cycles(M2)
                
                if bad_count2 == 0:
                    edges2 = get_edge_count(M2)
                    if edges2 > best_sa_edges:
                        best_sa_edges = edges2
                        best_sa_M = M2.copy()
        
        if best_sa_edges > best_edge_count:
            best_edge_count = best_sa_edges
            best_M = best_sa_M
    
    # === PHASE 3: Edge addition local search ===
    if best_M is not None and time.time() < deadline - 5.0:
        M = best_M.copy()
        current_edges = get_edge_count(M)
        
        # Try to add back missing edges
        missing = [i for i in range(100) if M[i] == -1]
        random.shuffle(missing)
        
        for flat_idx in missing:
            if time.time() > deadline - 5.0:
                break
            i, j = divmod(flat_idx, 10)
            
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for color in colors:
                M[flat_idx] = color
                bad = 0
                for (cyc_idx, pos) in cell_cycle_pos[(i, j)]:
                    c0 = M[cycle_edge_flat[cyc_idx, 0]]
                    c1 = M[cycle_edge_flat[cyc_idx, 1]]
                    c2 = M[cycle_edge_flat[cyc_idx, 2]]
                    c3 = M[cycle_edge_flat[cyc_idx, 3]]
                    if -1 in (c0, c1, c2, c3):
                        continue
                    if len(set([c0, c1, c2, c3])) < 4:
                        bad += 1
                        break
                if bad == 0:
                    current_edges += 1
                    break
                else:
                    M[flat_idx] = -1
        
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_M = M.copy()
    
    # Fallback: star graph
    if best_M is None:
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result
    
    # Convert to edge list
    result = []
    for i in range(10):
        for j in range(10):
            flat_idx = i * 10 + j
            if best_M[flat_idx] != -1:
                u = u_nodes[i]
                v = v_nodes[j]
                c = int(best_M[flat_idx])
                result.append([u, v, c])
    
    return result

# EVOLVE-BLOCK-END
