"""Dense 4-cycle rainbow edge-colouring on 20 vertices."""
import time
import random
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    n = 20
    num_colors = 20
    
    u_nodes = list(range(10))
    v_nodes = list(range(10, 20))
    
    # Precompute cycles
    cycle_list = []
    for i1 in range(10):
        for i2 in range(i1 + 1, 10):
            for j1 in range(10):
                for j2 in range(j1 + 1, 10):
                    cycle_list.append(((i1, j1), (i2, j1), (i2, j2), (i1, j2)))
    
    num_cycles = len(cycle_list)
    
    # Map cell to (cycle_idx, pos)
    cell_to_cycles = defaultdict(list)
    for idx, cyc in enumerate(cycle_list):
        for pos, (r, c) in enumerate(cyc):
            cell_to_cycles[(r, c)].append((idx, pos))
            
    def get_cycle_colors(cyc_idx, M):
        e0, e1, e2, e3 = cycle_list[cyc_idx]
        return (M[e0[0]*10 + e0[1]], M[e1[0]*10 + e1[1]], M[e2[0]*10 + e2[1]], M[e3[0]*10 + e3[1]])
        
    def is_bad(c0, c1, c2, c3):
        if c0 == -1 or c1 == -1 or c2 == -1 or c3 == -1:
            return False
        return len(set([c0, c1, c2, c3])) < 4

    best_M = None
    best_edge_count = 0
    
    random.seed(42)
    
    # SA Loop
    for restart in range(100):
        if time.time() > deadline - 15.0:
            break
            
        M = np.random.randint(0, num_colors, size=100, dtype=np.int32)
        bad_count = 0
        cycle_bad = np.zeros(num_cycles, dtype=bool)
        
        for idx in range(num_cycles):
            c0, c1, c2, c3 = get_cycle_colors(idx, M)
            if is_bad(c0, c1, c2, c3):
                bad_count += 1
                cycle_bad[idx] = True
                
        current_edges = 100
        T = 5.0
        cooling = 0.995
        local_best_bad = bad_count
        local_best_edges = current_edges
        local_best_M = M.copy()
        
        max_steps = 15000
        
        for step in range(max_steps):
            if time.time() > deadline - 15.0:
                break
                
            # Move Selection
            if random.random() < 0.8:
                # Recolor
                flat_idx = random.randint(0, 99)
                old_c = M[flat_idx]
                new_c = random.randint(0, num_colors - 1)
                if new_c == old_c: continue
                
                i, j = divmod(flat_idx, 10)
                affected = cell_to_cycles[(i, j)]
                old_statuses = []
                delta = 0
                
                M[flat_idx] = new_c
                
                for (cyc_idx, pos) in affected:
                    old_status = cycle_bad[cyc_idx]
                    old_statuses.append((cyc_idx, old_status))
                    
                    c0, c1, c2, c3 = get_cycle_colors(cyc_idx, M)
                    new_status = is_bad(c0, c1, c2, c3)
                    
                    if new_status != old_status:
                        if new_status: delta += 1
                        else: delta -= 1
                        cycle_bad[cyc_idx] = new_status
                        
                bad_count += delta
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    if bad_count < local_best_bad or (bad_count == local_best_bad and current_edges > local_best_edges):
                        local_best_bad = bad_count
                        local_best_edges = current_edges
                        local_best_M = M.copy()
                else:
                    # Revert
                    M[flat_idx] = old_c
                    bad_count -= delta
                    for (cyc_idx, old_status) in old_statuses:
                        cycle_bad[cyc_idx] = old_status
                        
            else:
                # Remove Edge
                flat_idx = random.randint(0, 99)
                if M[flat_idx] == -1: continue
                
                old_c = M[flat_idx]
                i, j = divmod(flat_idx, 10)
                affected = cell_to_cycles[(i, j)]
                old_statuses = []
                delta = 0
                
                M[flat_idx] = -1
                current_edges -= 1
                
                for (cyc_idx, pos) in affected:
                    old_status = cycle_bad[cyc_idx]
                    old_statuses.append((cyc_idx, old_status))
                    
                    c0, c1, c2, c3 = get_cycle_colors(cyc_idx, M)
                    new_status = is_bad(c0, c1, c2, c3)
                    
                    if new_status != old_status:
                        if new_status: delta += 1
                        else: delta -= 1
                        cycle_bad[cyc_idx] = new_status
                        
                bad_count += delta
                
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    if bad_count < local_best_bad or (bad_count == local_best_bad and current_edges > local_best_edges):
                        local_best_bad = bad_count
                        local_best_edges = current_edges
                        local_best_M = M.copy()
                else:
                    M[flat_idx] = old_c
                    current_edges += 1
                    bad_count -= delta
                    for (cyc_idx, old_status) in old_statuses:
                        cycle_bad[cyc_idx] = old_status
                        
            T *= cooling
            
        # End SA Restart
        if local_best_bad == 0 and local_best_edges > best_edge_count:
            best_edge_count = local_best_edges
            best_M = local_best_M.copy()
            
    # Phase 2: Add Edges to best_M if valid
    if best_M is not None:
        M = best_M.copy()
        bad_count = 0
        cycle_bad = np.zeros(num_cycles, dtype=bool)
        for idx in range(num_cycles):
            c0, c1, c2, c3 = get_cycle_colors(idx, M)
            if is_bad(c0, c1, c2, c3):
                bad_count += 1
                cycle_bad[idx] = True
                
        if bad_count == 0:
            # Try to add missing edges
            missing = [i for i in range(100) if M[i] == -1]
            random.shuffle(missing)
            for flat_idx in missing:
                if time.time() > deadline - 2.0: break
                i, j = divmod(flat_idx, 10)
                colors = list(range(num_colors))
                random.shuffle(colors)
                for c in colors:
                    M[flat_idx] = c
                    affected = cell_to_cycles[(i, j)]
                    delta = 0
                    old_statuses = []
                    for (cyc_idx, pos) in affected:
                        old_status = cycle_bad[cyc_idx]
                        old_statuses.append((cyc_idx, old_status))
                        c0, c1, c2, c3 = get_cycle_colors(cyc_idx, M)
                        new_status = is_bad(c0, c1, c2, c3)
                        if new_status != old_status:
                            if new_status: delta += 1
                            else: delta -= 1
                            cycle_bad[cyc_idx] = new_status
                            
                    bad_count += delta
                    if delta == 0:
                        break
                    else:
                        M[flat_idx] = -1
                        bad_count -= delta
                        for (cyc_idx, old_status) in old_statuses:
                            cycle_bad[cyc_idx] = old_status
                            
            if bad_count == 0:
                edges = np.sum(M != -1)
                if edges > best_edge_count:
                    best_edge_count = edges
                    best_M = M.copy()

    if best_M is None:
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result

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
