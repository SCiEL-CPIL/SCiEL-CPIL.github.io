import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 295.0

    # Base: Paley graph on 37 vertices
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    base_edges = [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
    base_n = q

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(n, adj):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    edges.append((i, j))
        return edges

    def find_violation(n, adj):
        """
        Returns a set of 5 vertices forming a K5 or Independent Set, or None if valid.
        """
        full_mask = (1 << n) - 1
        
        # Check K5
        for i in range(n):
            ni = adj[i]
            mask_j = ni
            while mask_j:
                lsb_j = mask_j & (-mask_j)
                j = lsb_j.bit_length() - 1
                mask_j ^= lsb_j
                
                common_jk = ni & adj[j]
                common_jk &= ~(1 << i) & ~(1 << j)
                
                mask_k = common_jk
                while mask_k:
                    lsb_k = mask_k & (-mask_k)
                    k = lsb_k.bit_length() - 1
                    mask_k ^= lsb_k
                    
                    common_ijk = common_jk & adj[k]
                    common_ijk &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                    
                    mask_l = common_ijk
                    while mask_l:
                        lsb_l = mask_l & (-mask_l)
                        l = lsb_l.bit_length() - 1
                        mask_l ^= lsb_l
                        
                        common_ijkl = common_ijk & adj[l]
                        common_ijkl &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                        
                        mask_m = common_ijkl
                        if mask_m:
                            # Found K5: i, j, k, l, and any m in mask_m
                            lsb_m = mask_m & (-mask_m)
                            m = lsb_m.bit_length() - 1
                            return {i, j, k, l, m}
        
        # Check Independent Set
        for i in range(n):
            nonni = full_mask & ~(adj[i] | (1 << i))
            mask_j = nonni
            while mask_j:
                lsb_j = mask_j & (-mask_j)
                j = lsb_j.bit_length() - 1
                mask_j ^= lsb_j
                
                nonnj = full_mask & ~(adj[j] | (1 << j))
                common_jk = nonni & nonnj
                common_jk &= ~(1 << i) & ~(1 << j)
                
                mask_k = common_jk
                while mask_k:
                    lsb_k = mask_k & (-mask_k)
                    k = lsb_k.bit_length() - 1
                    mask_k ^= lsb_k
                    
                    nonnk = full_mask & ~(adj[k] | (1 << k))
                    common_ijk = common_jk & nonnk
                    common_ijk &= ~(1 << i) & ~(1 << j) & ~(1 << k)
                    
                    mask_l = common_ijk
                    while mask_l:
                        lsb_l = mask_l & (-mask_l)
                        l = lsb_l.bit_length() - 1
                        mask_l ^= lsb_l
                        
                        nonnl = full_mask & ~(adj[l] | (1 << l))
                        common_ijkl = common_ijk & nonnl
                        common_ijkl &= ~(1 << i) & ~(1 << j) & ~(1 << k) & ~(1 << l)
                        
                        mask_m = common_ijkl
                        if mask_m:
                            lsb_m = mask_m & (-mask_m)
                            m = lsb_m.bit_length() - 1
                            return {i, j, k, l, m}
                        
        return None

    def local_search_targeted(n, adj, time_limit):
        current_adj = adj[:]
        stuck_count = 0
        last_violation = None
        
        while time.time() < time_limit:
            violation = find_violation(n, current_adj)
            if violation is None:
                return current_adj
            
            if violation == last_violation:
                stuck_count += 1
            else:
                stuck_count = 0
                last_violation = violation
            
            if stuck_count > 20:
                # Random perturbation
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    if u > v: u, v = v, u
                    current_adj[u] ^= (1 << v)
                    current_adj[v] ^= (1 << u)
                stuck_count = 0
                last_violation = None
                continue
            
            verts = list(violation)
            random.shuffle(verts)
            u, v = verts[0], verts[1]
            
            is_adjacent = (current_adj[u] >> v) & 1
            if is_adjacent:
                current_adj[u] &= ~(1 << v)
                current_adj[v] &= ~(1 << u)
            else:
                current_adj[u] |= (1 << v)
                current_adj[v] |= (1 << u)
                
        return None

    base_adj = build_adj(base_n, base_edges)
    
    # Try to grow
    for target_n in [38, 39, 40]:
        if time.time() > deadline - 30:
            break
            
        # Multiple attempts
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            init_adj = base_adj[:]
            init_adj.extend([0] * (target_n - base_n))
            
            # Connect new vertices
            # Target degree ~18
            for v in range(base_n, target_n):
                # Pick 18 random neighbors from existing + previous new
                candidates = list(range(v))
                random.shuffle(candidates)
                # Take first 18
                for u in candidates[:18]:
                    init_adj[u] |= (1 << v)
                    init_adj[v] |= (1 << u)
                    
            time_slice = min(60.0, deadline - time.time() - 5)
            if time_slice < 5:
                continue
                
            result = local_search_targeted(target_n, init_adj, time.time() + time_slice)
            
            if result is not None:
                edges = adj_to_edges(target_n, result)
                return target_n, edges
                
    return base_n, base_edges
