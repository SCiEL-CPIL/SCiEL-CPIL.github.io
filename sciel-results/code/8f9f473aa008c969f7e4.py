import random
import time
import math
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # Precompute Q3 masks in K8
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 1
            
    # Generate all Q3 masks
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
    
    # Precompute Lookup Table
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

    # Precompute Subsets and Edge Positions
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)
    
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    subset_edges = []
    
    for s_idx, combo in enumerate(subsets):
        edge_dict = {}
        for i in range(8):
            for j in range(i+1, 8):
                u = combo[i]
                v = combo[j]
                b_idx = bit_index_map[(i, j)]
                edge_dict[(u, v)] = b_idx
                edge_to_subsets[u][v].append((s_idx, b_idx))
                edge_to_subsets[v][u].append((s_idx, b_idx))
        subset_edges.append(edge_dict)
        
    if time.time() - start_time > time_limit:
        return []

    def compute_edge_count(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def greedy_fix(adj, induced_masks, violation_count, max_iters=800):
        """Remove edges to eliminate Q3 violations, preferring edges that unblock the most additions"""
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 8:
                break
            if violation_count == 0:
                break
            
            # Find a violating subset
            viol_subset = -1
            for s_idx in range(num_subsets):
                if lookup[induced_masks[s_idx]]:
                    viol_subset = s_idx
                    break
                    
            if viol_subset == -1:
                break
            
            combo = subsets[viol_subset]
            edge_dict = subset_edges[viol_subset]
            best_edge = None
            best_score = -999999
            
            for (u, v), b_idx in edge_dict.items():
                if not ((adj[u] >> v) & 1):
                    continue
                    
                bit = 1 << b_idx
                old_mask = induced_masks[viol_subset]
                new_mask = old_mask ^ bit
                
                delta_viol = -lookup[old_mask] + lookup[new_mask]
                # Count how many other subsets this edge affects
                num_affected = len(edge_to_subsets[u][v])
                # Score: prefer removing edges that reduce violations and affect fewer subsets
                score = -delta_viol * 1000 - num_affected * 10
                
                if score > best_score:
                    best_score = score
                    best_edge = (u, v, b_idx, viol_subset)
                    
            if best_edge is None:
                # Random fallback
                u = combo[random.randint(0, 7)]
                v = combo[random.randint(0, 7)]
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    b_idx = edge_dict.get((u, v))
                    if b_idx is not None:
                        bit = 1 << b_idx
                        for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                            old_m = induced_masks[s_idx2]
                            new_m = old_m ^ (1 << b_idx2)
                            violation_count += -lookup[old_m] + lookup[new_m]
                            induced_masks[s_idx2] = new_m
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
            else:
                u, v, b_idx, s_idx = best_edge
                bit = 1 << b_idx
                for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                    old_m = induced_masks[s_idx2]
                    new_m = old_m ^ (1 << b_idx2)
                    violation_count += -lookup[old_m] + lookup[new_m]
                    induced_masks[s_idx2] = new_m
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
        return adj, induced_masks, violation_count

    def local_search(adj, induced_masks, violation_count, time_budget):
        """Simulated annealing to maximize edges while keeping Q3-free"""
        current_edge_count = compute_edge_count(adj)
        best_edge_count = current_edge_count
        best_adj = adj[:]
        best_violation = violation_count
        
        T = 5.0
        cooling = 0.9998
        min_T = 0.005
        
        while time.time() - start_time < time_limit - 4:
            if T < min_T:
                T = min_T
                
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
                
            affected = edge_to_subsets[u][v]
            
            delta_violation = 0
            changes = []
            
            for (s_idx, b_idx) in affected:
                bit = 1 << b_idx
                old_mask = induced_masks[s_idx]
                new_mask = old_mask ^ bit
                changes.append((s_idx, new_mask))
                
                old_v = lookup[old_mask]
                new_v = lookup[new_mask]
                delta_violation += (new_v - old_v)
                
            edge_exists = (adj[u] >> v) & 1
            delta_edges = -1 if edge_exists else 1
            
            # Acceptance: strongly prefer edge additions when violations are equal
            if delta_violation < 0:
                accept = True
            elif delta_violation == 0:
                if delta_edges > 0:
                    accept = True
                else:
                    accept = random.random() < 0.03
            else:
                # Penalize violations heavily but allow some exploration
                cost = delta_violation * 200 - delta_edges * 5
                if cost <= 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(-cost / T)
                    
            if accept:
                for (s_idx, new_mask) in changes:
                    induced_masks[s_idx] = new_mask
                violation_count += delta_violation
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                current_edge_count += delta_edges
                
                if violation_count == 0 and current_edge_count > best_edge_count:
                    best_edge_count = current_edge_count
                    best_adj = adj[:]
                    best_violation = violation_count
                elif violation_count < best_violation:
                    best_violation = violation_count
                    if current_edge_count > best_edge_count:
                        best_edge_count = current_edge_count
                        best_adj = adj[:]
                        
            T *= cooling
            
        return best_adj, best_edge_count

    def final_polish(adj, masks, edge_count):
        """Multiple sweeps of greedy edge addition to reach local optimum"""
        changed = True
        sweep = 0
        while changed and time.time() - start_time < time_limit - 2 and sweep < 20:
            changed = False
            sweep += 1
            # Randomize order each sweep
            edge_list = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj[i] >> j) & 1)]
            random.shuffle(edge_list)
            for i, j in edge_list:
                if (adj[i] >> j) & 1:
                    continue
                affected = edge_to_subsets[i][j]
                creates_q3 = False
                for (s_idx, b_idx) in affected:
                    bit = 1 << b_idx
                    new_mask = masks[s_idx] | bit
                    if lookup[new_mask]:
                        creates_q3 = True
                        break
                if not creates_q3:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edge_count += 1
                    for (s_idx, b_idx) in affected:
                        masks[s_idx] |= (1 << b_idx)
                    changed = True
        return adj, edge_count

    # Multi-start approach with different initializations
    best_overall_adj = None
    best_overall_edges = 0
    
    # Strategy 1: Start from K16, greedy fix
    adj1 = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
            
    masks1 = [0] * num_subsets
    for s_idx in range(num_subsets):
        mask = 0
        edge_dict = subset_edges[s_idx]
        for (u, v), b_idx in edge_dict.items():
            if (adj1[u] >> v) & 1:
                mask |= (1 << b_idx)
        masks1[s_idx] = mask
        
    viol1 = sum(1 for m in masks1 if lookup[m])
    
    adj1, masks1, viol1 = greedy_fix(adj1, masks1, viol1)
    
    if viol1 == 0:
        adj1, edges1 = local_search(adj1, masks1, viol1, 25)
        if edges1 > best_overall_edges:
            best_overall_edges = edges1
            best_overall_adj = adj1[:]
    
    # Strategy 2: Random dense graph with varying densities
    for restart in range(4):
        if time.time() - start_time > time_limit - 15:
            break
            
        density = 0.6 + 0.15 * random.random()
        adj2 = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    adj2[i] |= (1 << j)
                    adj2[j] |= (1 << i)
                    
        masks2 = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (adj2[u] >> v) & 1:
                    mask |= (1 << b_idx)
            masks2[s_idx] = mask
            
        viol2 = sum(1 for m in masks2 if lookup[m])
        
        adj2, masks2, viol2 = greedy_fix(adj2, masks2, viol2)
        
        if viol2 == 0:
            adj2, edges2 = local_search(adj2, masks2, viol2, 15)
            if edges2 > best_overall_edges:
                best_overall_edges = edges2
                best_overall_adj = adj2[:]
    
    # Strategy 3: Structured initialization - complement of random matching
    for restart in range(3):
        if time.time() - start_time > time_limit - 10:
            break
            
        adj3 = [0] * n
        perm = list(range(n))
        random.shuffle(perm)
        matching = set()
        for i in range(n // 2):
            u = perm[2*i]
            v = perm[2*i+1]
            matching.add((min(u,v), max(u,v)))
            
        for i in range(n):
            for j in range(i+1, n):
                if (i, j) not in matching:
                    adj3[i] |= (1 << j)
                    adj3[j] |= (1 << i)
                    
        masks3 = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (adj3[u] >> v) & 1:
                    mask |= (1 << b_idx)
            masks3[s_idx] = mask
            
        viol3 = sum(1 for m in masks3 if lookup[m])
        
        adj3, masks3, viol3 = greedy_fix(adj3, masks3, viol3)
        
        if viol3 == 0:
            adj3, edges3 = local_search(adj3, masks3, viol3, 12)
            if edges3 > best_overall_edges:
                best_overall_edges = edges3
                best_overall_adj = adj3[:]
    
    # Strategy 4: Turan-like bipartite initialization
    if time.time() - start_time < time_limit - 12:
        adj4 = [0] * n
        # Split into two parts of 8
        part_a = list(range(8))
        part_b = list(range(8, 16))
        # Add all edges between parts except a few random ones
        for u in part_a:
            for v in part_b:
                if random.random() < 0.85:
                    adj4[u] |= (1 << v)
                    adj4[v] |= (1 << u)
        # Add some edges within parts
        for part in [part_a, part_b]:
            for i in range(len(part)):
                for j in range(i+1, len(part)):
                    if random.random() < 0.3:
                        u = part[i]
                        v = part[j]
                        adj4[u] |= (1 << v)
                        adj4[v] |= (1 << u)
                        
        masks4 = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (adj4[u] >> v) & 1:
                    mask |= (1 << b_idx)
            masks4[s_idx] = mask
            
        viol4 = sum(1 for m in masks4 if lookup[m])
        
        adj4, masks4, viol4 = greedy_fix(adj4, masks4, viol4)
        
        if viol4 == 0:
            adj4, edges4 = local_search(adj4, masks4, viol4, 12)
            if edges4 > best_overall_edges:
                best_overall_edges = edges4
                best_overall_adj = adj4[:]
    
    # Final polish with multiple sweeps
    if best_overall_adj is not None:
        final_masks = [0] * num_subsets
        for s_idx in range(num_subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (best_overall_adj[u] >> v) & 1:
                    mask |= (1 << b_idx)
            final_masks[s_idx] = mask
            
        best_overall_adj, best_overall_edges = final_polish(best_overall_adj, final_masks, best_overall_edges)
    
    # Return edges
    if best_overall_adj is None:
        return []
        
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_overall_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
