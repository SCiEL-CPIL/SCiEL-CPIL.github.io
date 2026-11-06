import random
import time
import math
import itertools

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    n = 16
    full_mask = (1 << 28) - 1
    
    # 1. Precompute Q3 masks in K8
    q3_edges_local = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    
    bit_index_map = {}
    idx = 0
    for i in range(8):
        for j in range(i+1, 8):
            bit_index_map[(i, j)] = idx
            idx += 28
            
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
    num_q3 = len(q3_masks_list)
    
    if time.time() - start_time > time_limit:
        return []

    # 2. Precompute Subsets and Edge Positions
    subsets = list(itertools.combinations(range(n), 8))
    num_subsets = len(subsets)
    
    # For each edge (u,v), store list of (subset_idx, bit_idx)
    edge_to_subsets = [[[] for _ in range(n)] for _ in range(n)]
    
    # For each subset, store edge mapping
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

    def is_q3_free(mask):
        """Check if a mask is Q3-free using precomputed Q3 masks."""
        for qm in q3_masks_list:
            if (mask & qm) == qm:
                return False
        return True

    def count_violations_in_mask(mask):
        """Count number of Q3 subgraphs in a mask."""
        count = 0
        for qm in q3_masks_list:
            if (mask & qm) == qm:
                count += 1
        return count

    def compute_induced_masks(adj):
        """Compute induced masks for all subsets."""
        masks = [0] * num_subsets
        for s_idx, combo in enumerate(subsets):
            mask = 0
            edge_dict = subset_edges[s_idx]
            for (u, v), b_idx in edge_dict.items():
                if (adj[u] >> v) & 1:
                    mask |= (1 << b_idx)
            masks[s_idx] = mask
        return masks

    def count_total_violations(masks):
        """Count total violations across all subsets."""
        total = 0
        for m in masks:
            if m != 0:
                total += count_violations_in_mask(m)
        return total

    def edge_count(adj):
        """Count edges in adjacency list."""
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # 3. Multi-restart greedy initialization
    best_adj = None
    best_edge_count = -1
    best_violation = float('inf')
    best_induced_masks = None
    
    num_greedy_restarts = 8
    
    for restart in range(num_greedy_restarts):
        if time.time() - start_time > time_limit - 80:
            break
            
        # Start with K16
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
        # Add some random edge removals for diversity
        if restart > 0:
            num_removals = random.randint(10, 25)
            for _ in range(num_removals):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v:
                    if u > v:
                        u, v = v, u
                    if (adj[u] >> v) & 1:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        
        # Compute initial induced masks
        induced_masks = compute_induced_masks(adj)
        violation_count = count_total_violations(induced_masks)
        cur_edge_count = edge_count(adj)
        
        # Greedy edge removal to eliminate violations
        for _ in range(300):
            if violation_count == 0:
                break
            if time.time() - start_time > time_limit - 80:
                break
                
            # Find a violating subset
            viol_subset = -1
            for s_idx in range(num_subsets):
                if count_violations_in_mask(induced_masks[s_idx]) > 0:
                    viol_subset = s_idx
                    break
                    
            if viol_subset == -1:
                break
                
            # Try removing each edge in this subset
            combo = subsets[viol_subset]
            edge_dict = subset_edges[viol_subset]
            best_edge = None
            best_delta = 0
            
            for (u, v), b_idx in edge_dict.items():
                if not ((adj[u] >> v) & 1):
                    continue
                    
                # Try removing this edge - count how many violations it fixes across all subsets
                total_delta = 0
                for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                    old_m = induced_masks[s_idx2]
                    if old_m == 0:
                        continue
                    new_m = old_m ^ (1 << b_idx2)
                    old_v = count_violations_in_mask(old_m)
                    new_v = count_violations_in_mask(new_m)
                    total_delta += (new_v - old_v)
                    
                if total_delta < best_delta:
                    best_delta = total_delta
                    best_edge = (u, v)
                    
            if best_edge is None:
                u = combo[random.randint(0, 7)]
                v = combo[random.randint(0, 7)]
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                if (adj[u] >> v) & 1:
                    for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                        old_m = induced_masks[s_idx2]
                        new_m = old_m ^ (1 << b_idx2)
                        induced_masks[s_idx2] = new_m
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    cur_edge_count -= 1
            else:
                u, v = best_edge
                for (s_idx2, b_idx2) in edge_to_subsets[u][v]:
                    old_m = induced_masks[s_idx2]
                    new_m = old_m ^ (1 << b_idx2)
                    induced_masks[s_idx2] = new_m
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                cur_edge_count -= 1
                violation_count += best_delta
                
        # Track best
        if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
            best_violation = violation_count
            best_adj = adj[:]
            best_edge_count = cur_edge_count
            best_induced_masks = induced_masks[:]

    # 4. Simulated Annealing from best greedy result
    adj = best_adj[:]
    induced_masks = best_induced_masks[:]
    violation_count = best_violation
    cur_edge_count = best_edge_count
    
    T = 1.0
    cooling = 0.9995
    min_T = 0.05
    no_improve_count = 0
    restart_threshold = 5000
    sa_start_time = time.time()
    
    while time.time() - start_time < time_limit - 15:
        if T < min_T:
            T = min_T
            
        # Pick random edge
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
            
            old_v = count_violations_in_mask(old_mask)
            new_v = count_violations_in_mask(new_mask)
            delta_violation += (new_v - old_v)
            
        # Calculate edge count change
        edge_exists = (adj[u] >> v) & 1
        delta_edges = -1 if edge_exists else 1
        
        # Acceptance: use a combined score
        # We want to minimize violations and maximize edges
        # Score = violations * large_weight - edges
        weight = 100
        old_score = violation_count * weight - cur_edge_count
        new_score = (violation_count + delta_violation) * weight - (cur_edge_count + delta_edges)
        delta_score = new_score - old_score
        
        if delta_score <= 0:
            accept = True
        else:
            accept = random.random() < math.exp(-delta_score / T)
                
        if accept:
            for (s_idx, new_mask) in changes:
                induced_masks[s_idx] = new_mask
            violation_count += delta_violation
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            cur_edge_count += delta_edges
            
            if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
                best_violation = violation_count
                best_adj = adj[:]
                best_edge_count = cur_edge_count
                best_induced_masks = induced_masks[:]
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        T *= cooling
        
        # Periodic restart if stuck
        if no_improve_count > restart_threshold:
            adj = best_adj[:]
            induced_masks = best_induced_masks[:]
            violation_count = best_violation
            cur_edge_count = best_edge_count
            T = 0.5
            no_improve_count = 0
            restart_threshold = max(2000, int(restart_threshold * 0.8))

    # 5. Final local search polish
    if time.time() - start_time < time_limit - 5:
        adj = best_adj[:]
        induced_masks = best_induced_masks[:]
        violation_count = best_violation
        cur_edge_count = best_edge_count
        
        # Try adding each missing edge
        improved = True
        while improved and time.time() - start_time < time_limit - 5:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if (adj[i] >> j) & 1:
                        continue  # Edge exists
                    
                    # Check if adding this edge creates violations
                    delta_v = 0
                    changes = []
                    for (s_idx, b_idx) in edge_to_subsets[i][j]:
                        bit = 1 << b_idx
                        old_mask = induced_masks[s_idx]
                        new_mask = old_mask | bit  # Adding edge
                        changes.append((s_idx, new_mask))
                        old_v = count_violations_in_mask(old_mask)
                        new_v = count_violations_in_mask(new_mask)
                        delta_v += (new_v - old_v)
                    
                    if delta_v <= 0:
                        # Apply the change
                        for (s_idx, new_mask) in changes:
                            induced_masks[s_idx] = new_mask
                        violation_count += delta_v
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
                        cur_edge_count += 1
                        improved = True
                        
                        if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
                            best_violation = violation_count
                            best_adj = adj[:]
                            best_edge_count = cur_edge_count
                            best_induced_masks = induced_masks[:]
        
        # Also try removing edges that help
        if time.time() - start_time < time_limit - 5:
            adj = best_adj[:]
            induced_masks = best_induced_masks[:]
            violation_count = best_violation
            cur_edge_count = best_edge_count
            
            improved = True
            while improved and time.time() - start_time < time_limit - 5:
                improved = False
                for i in range(n):
                    for j in range(i+1, n):
                        if not ((adj[i] >> j) & 1):
                            continue  # Edge doesn't exist
                        
                        # Check if removing this edge helps
                        delta_v = 0
                        changes = []
                        for (s_idx, b_idx) in edge_to_subsets[i][j]:
                            bit = 1 << b_idx
                            old_mask = induced_masks[s_idx]
                            new_mask = old_mask ^ bit  # Removing edge
                            changes.append((s_idx, new_mask))
                            old_v = count_violations_in_mask(old_mask)
                            new_v = count_violations_in_mask(new_mask)
                            delta_v += (new_v - old_v)
                        
                        if delta_v < 0:
                            # Apply the change
                            for (s_idx, new_mask) in changes:
                                induced_masks[s_idx] = new_mask
                            violation_count += delta_v
                            adj[i] ^= (1 << j)
                            adj[j] ^= (1 << i)
                            cur_edge_count -= 1
                            improved = True
                            
                            if violation_count < best_violation or (violation_count == best_violation and cur_edge_count > best_edge_count):
                                best_violation = violation_count
                                best_adj = adj[:]
                                best_edge_count = cur_edge_count
                                best_induced_masks = induced_masks[:]
    
    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
