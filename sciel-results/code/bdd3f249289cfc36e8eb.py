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
    
    # 2. Precompute Lookup Table - check if a mask contains ANY Q3 as subgraph
    # Instead of marking all supersets (too slow), we'll use a different approach:
    # For each 8-vertex subset, we check if its induced subgraph contains Q3
    # We'll precompute for each possible 28-bit mask whether it contains Q3
    # But that's 2^28 entries which is too much. Instead, we'll check on-the-fly
    # using a smarter method: for each 8-subset, we can precompute which Q3 patterns
    # are possible given the edge set.
    
    # Actually, let's use a hybrid: precompute a lookup for smaller masks or use
    # a different strategy. Given time constraints, let's use a faster method:
    # For each 8-vertex combo, we have 28 edges. We can precompute for each of the
    # 840 Q3 patterns whether it's a subgraph of the current induced edges.
    # But doing this per-step is expensive.
    
    # Better approach: Use the fact that Q3 has 12 edges. If our induced subgraph
    # on 8 vertices has < 12 edges, it can't contain Q3. So we can quickly reject.
    # For masks with >= 12 edges, we need to check more carefully.
    
    # Let's precompute a lookup table but only for masks that might be relevant.
    # Actually, let's just build the lookup table more efficiently using numpy.
    
    # Alternative: Build lookup by checking each mask against all Q3 patterns
    # This is O(2^28 * 840) which is too slow.
    
    # Best approach given constraints: Use a probabilistic/quick-check method
    # during the search, and only do full verification at the end.
    
    # For now, let's use a simpler but correct approach:
    # Precompute which bit positions correspond to each Q3 edge pattern
    # Then for a given mask, check if any Q3 pattern is fully contained.
    
    # Since we're doing simulated annealing, we need fast checks.
    # Let's precompute for each of the 840 Q3 masks, the 12 bit positions it uses.
    q3_bit_positions = []
    for qm in q3_masks_list:
        bits = []
        for b in range(28):
            if qm & (1 << b):
                bits.append(b)
        q3_bit_positions.append(bits)
    
    # lookup[mask] = True if mask contains any Q3
    # We'll use a bytearray of size 2^28 = 268M bytes ~ 256MB
    # This might be too much. Let's check memory.
    # Actually 2^28 = 268,435,456 bytes = 256 MB. This is borderline.
    # Let's try to build it efficiently.
    
    if time.time() - start_time > 30:
        # If we've spent too much time, use a fallback approach
        lookup = None
    else:
        # Try to build the lookup table
        lookup = bytearray(1 << 28)
        
        # For each Q3 mask, mark all its supersets
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

    # 4. Initialize Graph with a better starting point
    # Start with a random dense graph and try to remove violating edges
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.7:
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
    if lookup is not None:
        for mask in induced_masks:
            if lookup[mask]:
                violation_count += 1
    else:
        # Fallback: count violations by checking Q3 patterns directly
        for idx, combo in enumerate(subsets):
            mask = induced_masks[idx]
            edge_count = bin(mask).count('1')
            if edge_count >= 12:
                # Check if any Q3 pattern is contained
                for qbits in q3_bit_positions:
                    contained = True
                    for b in qbits:
                        if not (mask & (1 << b)):
                            contained = False
                            break
                    if contained:
                        violation_count += 1
                        break
                        
    best_violation = violation_count
    best_adj = adj[:]
    best_edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
    
    # 5. Simulated Annealing with smooth penalty
    T = 5.0
    cooling = 0.9995
    min_T = 0.01
    
    # Use a faster edge count update
    current_edge_count = best_edge_count
    non_improving_steps = 0
    restart_threshold = 5000
    
    while time.time() - start_time < time_limit:
        if T < min_T:
            T = min_T
            
        # Temperature restart when stuck
        if non_improving_steps > restart_threshold:
            T = max(T, 2.0)
            non_improving_steps = 0
            
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
            
            if lookup is not None:
                old_v = lookup[old_mask]
                new_v = lookup[new_mask]
            else:
                # Fallback check
                def has_q3(m):
                    ec = bin(m).count('1')
                    if ec < 12:
                        return 0
                    for qbits in q3_bit_positions:
                        contained = True
                        for b in qbits:
                            if not (m & (1 << b)):
                                contained = False
                                break
                        if contained:
                            return 1
                    return 0
                old_v = has_q3(old_mask)
                new_v = has_q3(new_mask)
            
            delta_violation += (new_v - old_v)
            
        # Calculate edge count change
        edge_exists = (adj[u] >> v) & 1
        delta_edges = -1 if edge_exists else 1
        
        # Smooth penalty: use log-sum-exp style soft penalty
        # Penalty = violation_count * penalty_weight - edge_count
        # Accept with probability based on delta of this score
        penalty_weight = 10000
        delta_score = delta_violation * penalty_weight - delta_edges
        
        # Accept if score improves, or with probability exp(-delta_score/T)
        if delta_score < 0:
            accept = True
        else:
            # Use smooth acceptance
            if T > 0:
                prob = math.exp(-delta_score / T)
            else:
                prob = 0.0
            accept = random.random() < prob
                
        if accept:
            # Apply changes
            for idx, new_mask in new_masks:
                induced_masks[idx] = new_mask
            violation_count += delta_violation
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edge_count += delta_edges
            
            # Update best
            if violation_count < best_violation or (violation_count == best_violation and current_edge_count > best_edge_count):
                best_violation = violation_count
                best_adj = adj[:]
                best_edge_count = current_edge_count
                non_improving_steps = 0
            else:
                non_improving_steps += 1
                
        T *= cooling
        
    # Final greedy pass: try to add edges that don't create violations
    for _ in range(8):
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                if not (best_adj[i] >> j) & 1:
                    # Try adding this edge
                    affected = subsets_containing_edge[i][j]
                    new_violations = 0
                    for idx in affected:
                        info = subset_info[idx]
                        b_idx = info[(i, j)]
                        bit = 1 << b_idx
                        old_mask = induced_masks[idx]
                        new_mask = old_mask | bit
                        if lookup is not None:
                            if lookup[new_mask] and not lookup[old_mask]:
                                new_violations += 1
                        else:
                            def has_q3(m):
                                ec = bin(m).count('1')
                                if ec < 12:
                                    return 0
                                for qbits in q3_bit_positions:
                                    contained = True
                                    for b in qbits:
                                        if not (m & (1 << b)):
                                            contained = False
                                            break
                                    if contained:
                                        return 1
                                return 0
                            if has_q3(new_mask) and not has_q3(old_mask):
                                new_violations += 1
                    if new_violations == 0:
                        # Add the edge
                        best_adj[i] |= (1 << j)
                        best_adj[j] |= (1 << i)
                        best_edge_count += 1
                        # Update masks
                        for idx in affected:
                            info = subset_info[idx]
                            b_idx = info[(i, j)]
                            induced_masks[idx] |= (1 << b_idx)
                        improved = True
        if not improved:
            break
    
    # Return best graph
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])
                
    return edges
