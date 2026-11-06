import random
import time
import math
import itertools
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 118.0  # Safety margin for 120s limit (150s hard kill)
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # 1. Precompute Q3 masks in K8
    # Q3 edges in 0..7:
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    # Map local edge (u, v) to bit index in 28-bit mask
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    # Generate all 840 Q3 masks
    q3_masks = set()
    for perm in itertools.permutations(range(8)):
        mask = 0
        for u, v in q3_edges_local:
            pu = perm[u]
            pv = perm[v]
            if pu > pv:
                pu, pv = pv, pu
            mask |= (1 << bit_index_map[(pu, pv)])
        q3_masks.add(mask)
        
    q3_masks_list = list(q3_masks)
    
    # 2. Precompute Lookup Table
    # lookup[mask] = 1 if mask contains any Q3 mask
    lookup = bytearray(1 << 28)
    
    for qm in q3_masks_list:
        Z = full_mask ^ qm
        s = 0
        while True:
            lookup[qm | s] = 1
            if s == Z:
                break
            s = (s - Z) & Z
            
    if time.time() - start_time > time_limit:
        return []

    # 3. Precompute Subsets and Edge Positions
    subsets = []
    subsets_containing_edge = [[[] for _ in range(n)] for _ in range(n)]
    
    subset_info = [] # list of dicts: {(u,v): bit_idx}
    
    for combo in itertools.combinations(range(n), 8):
        subsets.append(combo)
        info = {}
        for i in range(8):
            for j in range(i+1, 8):
                u = combo[i]
                v = combo[j]
                b_idx = bit_index_map[(i, j)]
                info[(u, v)] = b_idx
                info[(v, u)] = b_idx
                subsets_containing_edge[u][v].append(len(subsets) - 1)
                subsets_containing_edge[v][u].append(len(subsets) - 1)
        subset_info.append(info)
        
    if time.time() - start_time > time_limit:
        return []

    # 4. Greedy initialization: start dense, remove edges to eliminate Q3s
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    
    # Compute initial induced masks
    induced_masks = [0] * len(subsets)
    for idx, combo in enumerate(subsets):
        mask = 0
        info = subset_info[idx]
        for i in range(8):
            u = combo[i]
            for j in range(i+1, 8):
                v = combo[j]
                if (adj[u] >> v) & 1:
                    b_idx = info[(u, v)]
                    mask |= (1 << b_idx)
        induced_masks[idx] = mask
        
    # Compute initial violation count
    violation_count = 0
    for mask in induced_masks:
        if lookup[mask]:
            violation_count += 1
            
    best_violation = violation_count
    best_adj = adj[:]
    best_edge_count = sum(adj[i].bit_count() for i in range(n)) // 2
    
    # 5. Tabu search with QUBO-style penalty
    # Penalty weight for each Q3 violation
    penalty_weight = 50
    tabu_list = {}  # (u,v) -> step when it becomes non-tabu
    tabu_tenure = 7
    current_step = 0
    current_edge_count = best_edge_count
    
    # Use a better cooling schedule for simulated annealing
    T = 5.0
    cooling = 0.9998
    min_T = 0.005
    
    # Track best solution
    best_score = -violation_count * penalty_weight + best_edge_count
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            
        # Pick random edge
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        affected_indices = subsets_containing_edge[u][v]
        
        delta_violation = 0
        new_masks = []
        
        for idx in affected_indices:
            info = subset_info[idx]
            b_idx = info[(u, v)]
            bit = 1 << b_idx
            old_mask = induced_masks[idx]
            new_mask = old_mask ^ bit
            new_masks.append((idx, new_mask))
            
            old_v = lookup[old_mask]
            new_v = lookup[new_mask]
            delta_violation += (new_v - old_v)
            
        # Calculate edge count change
        edge_exists = (adj[u] >> v) & 1
        delta_edges = -1 if edge_exists else 1
        
        # Cost change: we want to minimize cost = violation_count * penalty_weight - edge_count
        # So delta_cost = penalty_weight * delta_violation - delta_edges
        # Accept if delta_cost < 0 (improvement) or with probability exp(-delta_cost / T)
        delta_cost = penalty_weight * delta_violation - delta_edges
        
        # Tabu check
        edge_key = (u, v)
        is_tabu = edge_key in tabu_list and tabu_list[edge_key] > current_step
        
        # Acceptance: accept if improves, or if not tabu and random accept
        if delta_cost < 0:
            accept = True
        elif not is_tabu and delta_cost < 2 * penalty_weight:
            # Allow some worsening moves to escape local minima
            accept = random.random() < math.exp(-delta_cost / T)
        else:
            accept = False
            
        if accept:
            # Apply changes
            for idx, new_mask in new_masks:
                induced_masks[idx] = new_mask
            violation_count += delta_violation
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edge_count += delta_edges
            current_step += 1
            
            # Add to tabu list
            tabu_list[edge_key] = current_step + tabu_tenure
            
            # Clean up old tabu entries
            if len(tabu_list) > 100:
                old_keys = [k for k, v in tabu_list.items() if v <= current_step]
                for k in old_keys:
                    del tabu_list[k]
            
            # Update best
            current_score = -violation_count * penalty_weight + current_edge_count
            if current_score > best_score:
                best_score = current_score
                best_violation = violation_count
                best_adj = adj[:]
                best_edge_count = current_edge_count
                # Reset temperature periodically when we find a new best
                if random.random() < 0.3:
                    T = max(T * 0.7, min_T)
                
        T *= cooling
        
    # Final greedy improvement: try adding edges that don't create Q3s
    # Use precomputed masks for efficiency
    final_masks = [0] * len(subsets)
    for idx, combo in enumerate(subsets):
        mask = 0
        info = subset_info[idx]
        for i in range(8):
            u = combo[i]
            for j in range(i+1, 8):
                v = combo[j]
                if (best_adj[u] >> v) & 1:
                    b_idx = info[(u, v)]
                    mask |= (1 << b_idx)
        final_masks[idx] = mask
    
    changed = True
    while changed:
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                if not ((best_adj[i] >> j) & 1):
                    # Check if adding this edge creates a Q3
                    affected = subsets_containing_edge[i][j]
                    creates_q3 = False
                    for idx in affected:
                        info = subset_info[idx]
                        b_idx = info[(i, j)]
                        bit = 1 << b_idx
                        new_mask = final_masks[idx] | bit
                        if lookup[new_mask]:
                            creates_q3 = True
                            break
                    if not creates_q3:
                        best_adj[i] |= (1 << j)
                        best_adj[j] |= (1 << i)
                        best_edge_count += 1
                        # Update all affected masks
                        for idx in affected:
                            info = subset_info[idx]
                            b_idx = info[(i, j)]
                            final_masks[idx] |= (1 << b_idx)
                        changed = True
                    
    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
