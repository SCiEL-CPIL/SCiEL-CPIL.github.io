import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin
    
    def precompute_subsets(n, k_min, k_max):
        """
        Precompute for each pair (u, v) and k, the list of masks of size k containing u and v.
        """
        subsets_by_pair = {}
        
        for k in range(k_min, k_max + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                # For every pair in comb, add this mask
                for i in range(k):
                    for j in range(i + 1, k):
                        u = comb[i]
                        v = comb[j]
                        key = (u, v) if u < v else (v, u)
                        if key not in subsets_by_pair:
                            subsets_by_pair[key] = {kk: [] for kk in range(k_min, k_max + 1)}
                        subsets_by_pair[key][k].append(mask)
                        
        return subsets_by_pair

    def is_regular_mask(mask, adj):
        """
        Check if induced subgraph on mask is regular.
        adj is a list of bitmasks.
        """
        m = mask
        verts = []
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            verts.append(v)
            m ^= lsb
            
        if not verts:
            return False
        
        d0 = (adj[verts[0]] & mask).bit_count()
        for v in verts[1:]:
            if (adj[v] & mask).bit_count() != d0:
                return False
        return True

    def count_violations(adj, n, k_min, k_max):
        """
        Count regular induced subgraphs of size k in [k_min, k_max].
        """
        count = 0
        for k in range(k_min, k_max + 1):
            for comb in itertools.combinations(range(n), k):
                mask = 0
                for v in comb:
                    mask |= (1 << v)
                if is_regular_mask(mask, adj):
                    count += 1
        return count

    def local_search(n, time_limit):
        """
        Local search to minimize number of regular induced subgraphs of size 6-10.
        """
        k_min = 6
        k_max = 10
        
        # Precompute subsets for fast delta calculation
        subsets_by_pair = precompute_subsets(n, k_min, k_max)
        
        # Initialize random graph
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if random.random() < 0.5:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        
        current_cost = count_violations(adj, n, k_min, k_max)
        
        best_cost = current_cost
        best_adj = adj[:]
        
        no_improve_counter = 0
        
        while time.time() < time_limit:
            if current_cost == 0:
                # Check full validity
                valid = True
                for k in range(k_max + 1, n + 1):
                    for comb in itertools.combinations(range(n), k):
                        mask = 0
                        for v in comb:
                            mask |= (1 << v)
                        if is_regular_mask(mask, adj):
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Reconstruct edges
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if adj[u] & (1 << v):
                                edges.append([u, v])
                    return {'n': n, 'edges': edges}
            
            # Pick a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            if u > v:
                u, v = v, u
                
            key = (u, v)
            if key not in subsets_by_pair:
                continue
                
            # Calculate delta cost
            delta = 0
            for k in range(k_min, k_max + 1):
                masks = subsets_by_pair[key].get(k, [])
                for mask in masks:
                    # Check if regular BEFORE
                    was_reg = is_regular_mask(mask, adj)
                    
                    has_edge = bool(adj[u] & (1 << v))
                    delta_deg = -1 if has_edge else 1
                    
                    deg_u = (adj[u] & mask).bit_count()
                    deg_v = (adj[v] & mask).bit_count()
                    
                    if was_reg:
                        delta -= 1
                    else:
                        # Check if it becomes regular
                        others_mask = mask & ~(1 << u) & ~(1 << v)
                        
                        if others_mask:
                            first_v = (others_mask & -others_mask).bit_length() - 1
                            target_deg = (adj[first_v] & mask).bit_count()
                            
                            is_const = True
                            m = others_mask & (others_mask - 1) # skip first
                            while m:
                                v2 = (m & -m).bit_length() - 1
                                if (adj[v2] & mask).bit_count() != target_deg:
                                    is_const = False
                                    break
                                m &= m - 1
                            
                            if is_const:
                                if target_deg == deg_u + delta_deg and target_deg == deg_v + delta_deg:
                                    delta += 1
                        else:
                            if deg_u + delta_deg == deg_v + delta_deg:
                                delta += 1

            # Accept move
            if delta < 0:
                # Flip edge
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj[:]
                    no_improve_counter = 0
                else:
                    no_improve_counter += 1
            elif delta == 0 and random.random() < 0.1:
                # Random walk
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                no_improve_counter += 1
            else:
                no_improve_counter += 1
                
            # Restart if stuck
            if no_improve_counter > 1000:
                adj = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                current_cost = count_violations(adj, n, k_min, k_max)
                no_improve_counter = 0

        return None

    # Try n=21
    t_limit_21 = start_time + 100
    g21 = local_search(21, t_limit_21)
    if g21:
        return g21

    # Try n=20
    t_limit_20 = start_time + 145
    g20 = local_search(20, t_limit_20)
    if g20:
        return g20

    # Fallback
    return {'n': 5, 'edges': []}
