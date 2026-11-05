"""Construct a C4-free graph with high minimum degree to certify R(C4,K1,39) > n."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    """
    Construct a C4-free graph with maximum n and minimum degree >= n-39.
    Uses bitsets for fast C4 checking and a greedy + local search approach.
    """
    
    def verify_c4_free(adj_bits, n):
        """Check if graph is C4-free using bitsets."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj_bits[u] >> v) & 1:
                    continue
                # Check for C4: u-v-x-y-u
                # x in N(v) \ {u}, y in N(u) \ {v}
                # If any x in N(v)\{u} is connected to any y in N(u)\{v}, C4 exists.
                
                sv = adj_bits[v] & ~(1 << u)
                su = adj_bits[u] & ~(1 << v)
                
                # Check if any node in sv is connected to any node in su
                temp_sv = sv
                while temp_sv:
                    lsb = temp_sv & -temp_sv
                    x = lsb.bit_length() - 1
                    if (adj_bits[x] & su):
                        return False
                    temp_sv ^= lsb
        return True

    def try_build(n, seed, min_deg_target):
        random.seed(seed)
        adj_bits = [0] * n
        degree = [0] * n
        
        def get_neighbors(u, exclude=-1):
            """Yield neighbors of u, optionally excluding one."""
            mask = adj_bits[u]
            if exclude >= 0:
                mask &= ~(1 << exclude)
            while mask:
                lsb = mask & -mask
                yield lsb.bit_length() - 1
                mask ^= lsb

        def can_add_edge(u, v):
            """Check if adding edge (u,v) creates a C4."""
            sv = adj_bits[v] & ~(1 << u)
            su = adj_bits[u] & ~(1 << v)
            
            if sv == 0 or su == 0:
                return True
                
            # Iterate over the smaller set
            count_sv = bin(sv).count('1')
            count_su = bin(su).count('1')
            
            if count_sv < count_su:
                temp = sv
                target = su
            else:
                temp = su
                target = sv
                
            while temp:
                lsb = temp & -temp
                node = lsb.bit_length() - 1
                if adj_bits[node] & target:
                    return False
                temp ^= lsb
            return True

        # Phase 1: Greedy
        max_rounds = 200
        for rnd in range(max_rounds):
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj_bits[u] >> v) & 1:
                        continue
                    if can_add_edge(u, v):
                        score = degree[u] + degree[v]
                        candidates.append((score, u, v))
            
            if not candidates:
                break
            
            candidates.sort()
            
            added = 0
            for score, u, v in candidates:
                if (adj_bits[u] >> v) & 1:
                    continue
                if can_add_edge(u, v):
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    degree[u] += 1
                    degree[v] += 1
                    added += 1
                    if added >= 10:
                        break
            
            if added == 0:
                break
        
        # Phase 2: Local Search (1-moves and 2-moves)
        # Try to improve min degree by swapping edges
        for step in range(5000):
            min_deg = min(degree)
            if min_deg >= min_deg_target:
                break
            
            low_verts = [i for i in range(n) if degree[i] < min_deg_target]
            if not low_verts:
                break
                
            improved = False
            random.shuffle(low_verts)
            
            # 1-move search
            for u in low_verts:
                neighbors_u = list(get_neighbors(u))
                if not neighbors_u:
                    continue
                    
                non_neighbors_u = [v for v in range(n) if v != u and not ((adj_bits[u] >> v) & 1)]
                
                # Sort non-neighbors by degree (low first)
                non_neighbors_u.sort(key=lambda x: degree[x])
                # Sort neighbors by degree (high first, so we remove from high degree nodes)
                neighbors_u.sort(key=lambda x: -degree[x])
                
                for z in non_neighbors_u[:10]:
                    if degree[z] >= min_deg_target:
                        continue
                        
                    for w in neighbors_u[:10]:
                        if degree[w] <= min_deg_target:
                            continue
                            
                        # Tentatively remove (u, w)
                        adj_bits[u] &= ~(1 << w)
                        adj_bits[w] &= ~(1 << u)
                        degree[u] -= 1
                        degree[w] -= 1
                        
                        # Try to add (u, z)
                        if can_add_edge(u, z):
                            adj_bits[u] |= (1 << z)
                            adj_bits[z] |= (1 << u)
                            degree[u] += 1
                            degree[z] += 1
                            improved = True
                            break
                        else:
                            # Revert
                            adj_bits[u] |= (1 << w)
                            adj_bits[w] |= (1 << u)
                            degree[u] += 1
                            degree[w] += 1
                            
                    if improved:
                        break
                        
                if improved:
                    break
            
            if improved:
                continue

            # 2-move search if 1-move failed
            # Pick two low vertices u, v
            if len(low_verts) >= 2:
                u, v = random.sample(low_verts, 2)
                neighbors_u = list(get_neighbors(u))
                neighbors_v = list(get_neighbors(v))
                
                # We want to swap edges: remove (u, w) and (v, x), add (u, x) and (v, w)
                # Conditions:
                # 1. w in N(u), x in N(v)
                # 2. (u, x) and (v, w) are not edges
                # 3. Adding (u, x) and (v, w) doesn't create C4
                # 4. degree[w] > min_deg_target and degree[x] > min_deg_target
                
                # To keep it fast, we limit the search space
                # Sort neighbors to prioritize high degree for removal
                neighbors_u.sort(key=lambda x: -degree[x])
                neighbors_v.sort(key=lambda x: -degree[x])
                
                found_2move = False
                for w in neighbors_u[:5]:
                    if degree[w] <= min_deg_target:
                        continue
                    for x in neighbors_v[:5]:
                        if degree[x] <= min_deg_target:
                            continue
                        if w == x:
                            continue
                        # Check if (u, x) and (v, w) are non-edges
                        if (adj_bits[u] >> x) & 1:
                            continue
                        if (adj_bits[v] >> w) & 1:
                            continue
                        
                        # Tentatively remove (u, w) and (v, x)
                        adj_bits[u] &= ~(1 << w)
                        adj_bits[w] &= ~(1 << u)
                        adj_bits[v] &= ~(1 << x)
                        adj_bits[x] &= ~(1 << v)
                        degree[u] -= 1
                        degree[w] -= 1
                        degree[v] -= 1
                        degree[x] -= 1
                        
                        # Check if we can add (u, x) and (v, w)
                        # We must check C4 for both additions
                        if can_add_edge(u, x) and can_add_edge(v, w):
                            # Apply
                            adj_bits[u] |= (1 << x)
                            adj_bits[x] |= (1 << u)
                            adj_bits[v] |= (1 << w)
                            adj_bits[w] |= (1 << v)
                            degree[u] += 1
                            degree[x] += 1
                            degree[v] += 1
                            degree[w] += 1
                            improved = True
                            found_2move = True
                            break
                        else:
                            # Revert
                            adj_bits[u] |= (1 << w)
                            adj_bits[w] |= (1 << u)
                            adj_bits[v] |= (1 << x)
                            adj_bits[x] |= (1 << v)
                            degree[u] += 1
                            degree[w] += 1
                            degree[v] += 1
                            degree[x] += 1
                    
                    if found_2move:
                        break
                
            if not improved:
                break

        # Verify
        min_deg = min(degree)
        if min_deg < min_deg_target:
            return None
        if not verify_c4_free(adj_bits, n):
            return None
        
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Try n=47 first, then 46, etc.
    for n in [47, 46, 45, 44, 43, 42, 41, 40, 39]:
        min_deg_target = n - 39
        if min_deg_target < 0:
            min_deg_target = 0
            
        start_time = time.time()
        seeds_tried = 0
        while time.time() - start_time < 110:
            result = try_build(n, seeds_tried, min_deg_target)
            if result is not None:
                return {'n': n, 'edges': result}
            seeds_tried += 1
            if seeds_tried > 100:
                break
    
    return {'n': 39, 'edges': []}

# EVOLVE-BLOCK-END
