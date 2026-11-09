"""Dense 4-cycle rainbow edge-colouring on 20 vertices."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n = 20
    num_colors = 20
    
    # Precompute all 4-cycles in K_20
    cycles = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    a, b, c, d = i, j, k, l
                    # Cycle 1: a-b, b-c, c-d, d-a
                    cycles.append(((a,b), (b,c), (c,d), (d,a)))
                    # Cycle 2: a-b, b-d, d-c, c-a
                    cycles.append(((a,b), (b,d), (d,c), (c,a)))
                    # Cycle 3: a-c, c-b, b-d, d-a
                    cycles.append(((a,c), (c,b), (b,d), (d,a)))
                    
    num_cycles = len(cycles)
    
    cyc_u = np.zeros((num_cycles, 4), dtype=np.int32)
    cyc_v = np.zeros((num_cycles, 4), dtype=np.int32)
    for idx, cyc in enumerate(cycles):
        for k in range(4):
            u, v = cyc[k]
            cyc_u[idx, k] = u
            cyc_v[idx, k] = v
            
    edge_to_cycles = {}
    for idx in range(num_cycles):
        for k in range(4):
            u, v = int(cyc_u[idx, k]), int(cyc_v[idx, k])
            if u > v:
                u, v = v, u
            key = u * n + v
            if key not in edge_to_cycles:
                edge_to_cycles[key] = []
            edge_to_cycles[key].append(idx)
            
    def get_bad_mask(M):
        c0 = M[cyc_u[:, 0], cyc_v[:, 0]]
        c1 = M[cyc_u[:, 1], cyc_v[:, 1]]
        c2 = M[cyc_u[:, 2], cyc_v[:, 2]]
        c3 = M[cyc_u[:, 3], cyc_v[:, 3]]
        
        all_exist = (c0 != -1) & (c1 != -1) & (c2 != -1) & (c3 != -1)
        
        distinct = (c0 != c1) & (c0 != c2) & (c0 != c3) & \
                   (c1 != c2) & (c1 != c3) & (c2 != c3)
                   
        bad = all_exist & ~distinct
        return bad
        
    def count_edges(M):
        return int(np.sum(M[np.triu_indices(n, k=1)] != -1))
        
    def edge_key(u, v):
        if u > v:
            u, v = v, u
        return u * n + v
        
    # Initialize M with random colors for all edges
    M = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        for j in range(i+1, n):
            c = random.randint(0, num_colors - 1)
            M[i, j] = c
            M[j, i] = c
    np.fill_diagonal(M, -1)
    
    # Best is empty graph
    best_M = np.full((n, n), -1, dtype=np.int32)
    best_edges = 0
    
    bad_mask = get_bad_mask(M)
    current_bad_count = np.sum(bad_mask)
    
    # Main Loop
    while time.time() < deadline - 5.0:
        if current_bad_count == 0:
            # Update best if better
            current_edges = count_edges(M)
            if current_edges > best_edges:
                best_edges = current_edges
                best_M = M.copy()
                
            # Try to add edges
            missing = []
            for i in range(n):
                for j in range(i+1, n):
                    if M[i, j] == -1:
                        missing.append((i, j))
                        
            if not missing:
                break # Complete and valid
                
            random.shuffle(missing)
            added = False
            for u, v in missing:
                if time.time() > deadline - 5.0:
                    break
                colors = list(range(num_colors))
                random.shuffle(colors)
                
                for c in colors:
                    M[u, v] = c
                    M[v, u] = c
                    
                    key = edge_key(u, v)
                    affected_cycles = edge_to_cycles.get(key, [])
                    
                    valid = True
                    for aidx in affected_cycles:
                        c0 = M[cyc_u[aidx, 0], cyc_v[aidx, 0]]
                        c1 = M[cyc_u[aidx, 1], cyc_v[aidx, 1]]
                        c2 = M[cyc_u[aidx, 2], cyc_v[aidx, 2]]
                        c3 = M[cyc_u[aidx, 3], cyc_v[aidx, 3]]
                        
                        if -1 in (c0, c1, c2, c3):
                            continue
                            
                        if len(set([c0, c1, c2, c3])) < 4:
                            valid = False
                            break
                            
                    if valid:
                        added = True
                        # Update best?
                        # We added an edge, so edge count increased.
                        # And it's valid.
                        new_edges = count_edges(M)
                        if new_edges > best_edges:
                            best_edges = new_edges
                            best_M = M.copy()
                        break
                    else:
                        M[u, v] = -1
                        M[v, u] = -1
                        
            if added:
                # State is still valid (we checked)
                # But we might have added an edge that allows more additions?
                # We just loop again.
                continue
                
        else:
            # Fix bad cycles
            bad_indices = np.where(bad_mask)[0]
            if len(bad_indices) == 0:
                current_bad_count = 0
                continue
                
            idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
            
            edges_in_cycle = []
            for k in range(4):
                u, v = int(cyc_u[idx, k]), int(cyc_v[idx, k])
                edges_in_cycle.append((u, v))
                
            fixed = False
            random.shuffle(edges_in_cycle)
            for u, v in edges_in_cycle:
                if time.time() > deadline - 5.0:
                    break
                if M[u, v] == -1:
                    continue
                    
                old_color = M[u, v]
                colors = list(range(num_colors))
                random.shuffle(colors)
                
                for c in colors:
                    if c == old_color:
                        continue
                        
                    other_colors = []
                    for (ou, ov) in edges_in_cycle:
                        if (ou == u and ov == v) or (ou == v and ov == u):
                            continue
                        other_colors.append(M[ou, ov])
                        
                    if c in other_colors:
                        continue
                        
                    M[u, v] = c
                    M[v, u] = c
                    
                    key = edge_key(u, v)
                    affected_cycles = edge_to_cycles.get(key, [])
                    
                    valid = True
                    for aidx in affected_cycles:
                        c0 = M[cyc_u[aidx, 0], cyc_v[aidx, 0]]
                        c1 = M[cyc_u[aidx, 1], cyc_v[aidx, 1]]
                        c2 = M[cyc_u[aidx, 2], cyc_v[aidx, 2]]
                        c3 = M[cyc_u[aidx, 3], cyc_v[aidx, 3]]
                        
                        if -1 in (c0, c1, c2, c3):
                            continue
                            
                        if len(set([c0, c1, c2, c3])) < 4:
                            valid = False
                            break
                            
                    if valid:
                        fixed = True
                        bad_mask = get_bad_mask(M)
                        current_bad_count = np.sum(bad_mask)
                        break
                    else:
                        M[u, v] = old_color
                        M[v, u] = old_color
                        
                if fixed:
                    break
                    
            if not fixed:
                u, v = edges_in_cycle[random.randint(0, 3)]
                M[u, v] = -1
                M[v, u] = -1
                bad_mask = get_bad_mask(M)
                current_bad_count = np.sum(bad_mask)
                
    # Final Polish
    if time.time() < deadline - 2.0:
        M = best_M.copy()
        current_edges = best_edges
        
        missing = []
        for i in range(n):
            for j in range(i+1, n):
                if M[i, j] == -1:
                    missing.append((i, j))
                    
        random.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 2.0:
                break
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for c in colors:
                M[u, v] = c
                M[v, u] = c
                
                key = edge_key(u, v)
                affected_cycles = edge_to_cycles.get(key, [])
                
                valid = True
                for aidx in affected_cycles:
                    c0 = M[cyc_u[aidx, 0], cyc_v[aidx, 0]]
                    c1 = M[cyc_u[aidx, 1], cyc_v[aidx, 1]]
                    c2 = M[cyc_u[aidx, 2], cyc_v[aidx, 2]]
                    c3 = M[cyc_u[aidx, 3], cyc_v[aidx, 3]]
                    
                    if -1 in (c0, c1, c2, c3):
                        continue
                        
                    if len(set([c0, c1, c2, c3])) < 4:
                        valid = False
                        break
                        
                if valid:
                    current_edges += 1
                    break
                else:
                    M[u, v] = -1
                    M[v, u] = -1
                    
        if current_edges > best_edges:
            best_edges = current_edges
            best_M = M.copy()
            
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if best_M[i, j] != -1:
                result.append([i, j, int(best_M[i, j])])
                
    return result
