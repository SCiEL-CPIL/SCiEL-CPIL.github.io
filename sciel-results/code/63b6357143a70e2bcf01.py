import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            for v in range(n):
                if adj[u, v]:
                    masks[u] |= (1 << v)
        return masks
    
    def build_comp_masks(n, masks):
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        return comp_masks
    
    def has_k4(n, masks):
        """Check if graph has any K4 using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if masks[wi] & (1 << xi):
                                return True
        return False
    
    def has_ind6(n, comp_masks):
        """Check if graph has any independent set of size 6 using bitmasks."""
        for u in range(n):
            for v in range(u + 1, n):
                if comp_masks[u] & (1 << v):
                    common = comp_masks[u] & comp_masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if comp_masks[wi] & (1 << xi):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                                common2 &= ~((1 << (xi + 1)) - 1)
                                if common2:
                                    return True
        return False
    
    def is_valid(n, masks, comp_masks):
        return not has_k4(n, masks) and not has_ind6(n, comp_masks)
    
    def count_violations(n, masks, comp_masks):
        """Count K4s and independent 6-sets."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if masks[u] & (1 << v):
                    common = masks[u] & masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if masks[wi] & (1 << xi):
                                k4_count += 1
        
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if comp_masks[u] & (1 << v):
                    common = comp_masks[u] & comp_masks[v]
                    common &= ~((1 << (v + 1)) - 1)
                    w_mask = common
                    while w_mask:
                        lsb_w = w_mask & -w_mask
                        wi = lsb_w.bit_length() - 1
                        w_mask ^= lsb_w
                        x_mask = common & ~((1 << (wi + 1)) - 1)
                        while x_mask:
                            lsb_x = x_mask & -x_mask
                            xi = lsb_x.bit_length() - 1
                            x_mask ^= lsb_x
                            if comp_masks[wi] & (1 << xi):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi] & comp_masks[xi]
                                common2 &= ~((1 << (xi + 1)) - 1)
                                if common2:
                                    ind6_count += 1
        return k4_count + ind6_count
    
    def spectral_init(n, seed_rng):
        """Generate a candidate graph using spectral methods.
        
        Strategy: Use random symmetric matrices, compute top eigenvectors,
        binarize to form adjacency with target edge density ~0.5.
        Also try structured constructions (Paley-like, random regular-ish).
        """
        adj = np.zeros((n, n), dtype=np.int8)
        
        # Method 1: Random symmetric with target density
        # For K4-free and alpha <= 5, density should be around 0.4-0.6
        p = 0.45 + 0.1 * seed_rng.random()
        for i in range(n):
            for j in range(i + 1, n):
                if seed_rng.random() < p:
                    adj[i, j] = 1
                    adj[j, i] = 1
        
        # Method 2: Spectral - use top eigenvector of random matrix
        # Create random symmetric matrix, take top eigenvector, threshold
        A = np.random.randn(n, n)
        A = (A + A.T) / 2.0
        # Remove diagonal
        np.fill_diagonal(A, 0)
        
        # Get top few eigenvectors
        try:
            evals, evecs = np.linalg.eigh(A)
            # Sort by absolute eigenvalue
            idx = np.argsort(-np.abs(evals))
            top_evecs = evecs[:, idx[:3]]  # Top 3 eigenvectors
            
            # Combine: use sign pattern of combined eigenvectors
            # Weight by eigenvalue magnitude
            weights = np.abs(evals[idx[:3]])
            weights /= (weights.sum() + 1e-10)
            combined = np.zeros(n)
            for k in range(3):
                combined += weights[k] * top_evecs[:, k]
            
            # Binarize: vertices with positive combined value connect to each other
            # This creates a bipartite-like structure that may avoid K4
            threshold = np.median(combined)
            pos_mask = combined > threshold
            neg_mask = ~pos_mask
            
            # Edges within positive set and within negative set
            # (bipartite edges would create independent sets)
            for i in range(n):
                for j in range(i + 1, n):
                    if pos_mask[i] and pos_mask[j]:
                        if seed_rng.random() < 0.6:
                            adj[i, j] = 1
                            adj[j, i] = 1
                    elif neg_mask[i] and neg_mask[j]:
                        if seed_rng.random() < 0.6:
                            adj[i, j] = 1
                            adj[j, i] = 1
        except:
            pass
        
        # Method 3: Structured - use quadratic residue (Paley-like) if n is prime
        if n > 2:
            # Check if n is prime
            is_prime = True
            for d in range(2, int(n**0.5) + 1):
                if n % d == 0:
                    is_prime = False
                    break
            if is_prime and n % 4 == 1:  # Paley requires p ≡ 1 mod 4
                squares = {x * x % n for x in range(1, n)}
                for i in range(n):
                    for j in range(i + 1, n):
                        if (j - i) % n in squares:
                            adj[i, j] = 1
                            adj[j, i] = 1
                return adj
        
        return adj
    
    def try_repair(n, time_budget, seed_adj=None):
        """Local search with targeted repair strategy."""
        end_time = time.time() + time_budget
        
        if seed_adj is not None and seed_adj.shape == (n, n):
            adj = seed_adj.copy()
        else:
            adj = spectral_init(n, random)
        
        masks = build_masks(n, adj)
        comp_masks = build_comp_masks(n, masks)
        current_cost = count_violations(n, masks, comp_masks)
        
        if current_cost == 0:
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges
        
        # Simulated annealing with incremental cost updates
        T = 3.0
        min_T = 0.01
        cooling = 0.99995
        no_improve = 0
        max_no_improve = 10000
        full_mask = (1 << n) - 1
        best_cost = current_cost
        best_adj = adj.copy()
        
        while time.time() < end_time and no_improve < max_no_improve:
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            old_state = adj[u, v]
            
            if old_state == 0:
                # Adding edge (u,v): new K4s formed, ind6s destroyed
                common_nv = masks[u] & masks[v]
                common_nv &= ~(1 << u)
                common_nv &= ~(1 << v)
                new_k4 = 0
                w_mask = common_nv
                while w_mask:
                    lsb_w = w_mask & -w_mask
                    wi = lsb_w.bit_length() - 1
                    w_mask ^= lsb_w
                    x_mask = common_nv & ~((1 << (wi + 1)) - 1)
                    while x_mask:
                        lsb_x = x_mask & -x_mask
                        xi = lsb_x.bit_length() - 1
                        x_mask ^= lsb_x
                        if masks[wi] & (1 << xi):
                            new_k4 += 1
                
                # Count ind6 destroyed
                destroyed_ind6 = 0
                if comp_masks[u] & (1 << v):
                    common_cv = comp_masks[u] & comp_masks[v]
                    common_cv &= ~((1 << (v + 1)) - 1)
                    w_mask2 = common_cv
                    while w_mask2:
                        lsb_w2 = w_mask2 & -w_mask2
                        wi2 = lsb_w2.bit_length() - 1
                        w_mask2 ^= lsb_w2
                        x_mask2 = common_cv & ~((1 << (wi2 + 1)) - 1)
                        while x_mask2:
                            lsb_x2 = x_mask2 & -x_mask2
                            xi2 = lsb_x2.bit_length() - 1
                            x_mask2 ^= lsb_x2
                            if comp_masks[wi2] & (1 << xi2):
                                common2 = comp_masks[u] & comp_masks[v] & comp_masks[wi2] & comp_masks[xi2]
                                common2 &= ~((1 << (xi2 + 1)) - 1)
                                if common2:
                                    destroyed_ind6 += 1
                
                delta = new_k4 - destroyed_ind6
            else:
                # Removing edge (u,v): K4s destroyed, new ind6s created
                destroyed_k4 = 0
                common_mv = masks[u] & masks[v]
                common_mv &= ~((1 << (v + 1)) - 1)
                w_mask3 = common_mv
                while w_mask3:
                    lsb_w3 = w_mask3 & -w_mask3
                    wi3 = lsb_w3.bit_length() - 1
                    w_mask3 ^= lsb_w3
                    x_mask3 = common_mv & ~((1 << (wi3 + 1)) - 1)
                    while x_mask3:
                        lsb_x3 = x_mask3 & -x_mask3
                        xi3 = lsb_x3.bit_length() - 1
                        x_mask3 ^= lsb_x3
                        if masks[wi3] & (1 << xi3):
                            destroyed_k4 += 1
                
                # New ind6 created
                new_ind6 = 0
                temp_comp_u = comp_masks[u] | (1 << v)
                temp_comp_v = comp_masks[v] | (1 << u)
                common_cv2 = temp_comp_u & temp_comp_v
                common_cv2 &= ~((1 << (v + 1)) - 1)
                w_mask4 = common_cv2
                while w_mask4:
                    lsb_w4 = w_mask4 & -w_mask4
                    wi4 = lsb_w4.bit_length() - 1
                    w_mask4 ^= lsb_w4
                    x_mask4 = common_cv2 & ~((1 << (wi4 + 1)) - 1)
                    while x_mask4:
                        lsb_x4 = x_mask4 & -x_mask4
                        xi4 = lsb_x4.bit_length() - 1
                        x_mask4 ^= lsb_x4
                        if (comp_masks[wi4] & (1 << xi4)) and (comp_masks[wi4] & (1 << u)) and (comp_masks[wi4] & (1 << v)) and (comp_masks[xi4] & (1 << u)) and (comp_masks[xi4] & (1 << v)):
                            new_ind6 += 1
                
                delta = -destroyed_k4 + new_ind6
            
            if delta < 0:
                # Accept improving move
                adj[u, v] = 1
                adj[v, u] = 1
                if old_state == 0:
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                    comp_masks[u] &= ~(1 << v)
                    comp_masks[v] &= ~(1 << u)
                else:
                    masks[u] &= ~(1 << v)
                    masks[v] &= ~(1 << u)
                    comp_masks[u] |= (1 << v)
                    comp_masks[v] |= (1 << u)
                current_cost += delta
                no_improve = 0
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_adj = adj.copy()
                if current_cost <= 0:
                    if is_valid(n, masks, comp_masks):
                        edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
                        return edges
            else:
                T = max(min_T, T * cooling)
                if random.random() < np.exp(-delta / T):
                    adj[u, v] = 1
                    adj[v, u] = 1
                    if old_state == 0:
                        masks[u] |= (1 << v)
                        masks[v] |= (1 << u)
                        comp_masks[u] &= ~(1 << v)
                        comp_masks[v] &= ~(1 << u)
                    else:
                        masks[u] &= ~(1 << v)
                        masks[v] &= ~(1 << u)
                        comp_masks[u] |= (1 << v)
                        comp_masks[v] |= (1 << u)
                    current_cost += delta
                    no_improve += 1
                else:
                    no_improve += 1
        
        if is_valid(n, masks, comp_masks):
            edges = [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]
            return edges
        if best_cost < current_cost:
            best_masks = build_masks(n, best_adj)
            best_comp = build_comp_masks(n, best_masks)
            if is_valid(n, best_masks, best_comp):
                edges = [[i, j] for i in range(n) for j in range(i + 1, n) if best_adj[i, j]]
                return edges
        return None
    
    # Build baseline: 17-vertex Paley graph
    squares = {x * x % 17 for x in range(1, 17)}
    adj17 = np.zeros((17, 17), dtype=np.int8)
    for i in range(17):
        for j in range(i + 1, 17):
            if (j - i) % 17 in squares:
                adj17[i, j] = 1
                adj17[j, i] = 1
    edges_17 = [[i, j] for i in range(17) for j in range(i + 1, 17) if adj17[i, j]]
    best_n = 17
    best_edges = edges_17
    
    # Phase 1: Try to jump directly to high n using spectral init
    # Try n = 35, 34, 33, 32, 31, 30 with multiple restarts
    remaining = deadline - time.time() - 3.0
    if remaining > 5.0:
        # Allocate time to high-n attempts
        high_n_budget = remaining * 0.6
        per_target = high_n_budget / 6.0  # 6 targets: 35,34,33,32,31,30
        
        for target_n in [35, 34, 33, 32, 31, 30]:
            if time.time() > deadline - 3.0:
                break
            if target_n <= best_n:
                continue
            
            budget = min(per_target, remaining - (time.time() - start_time) - 3.0)
            if budget < 1.0:
                break
            
            # Multiple restarts for this target
            num_restarts = max(1, int(budget / 2.0))
            found = None
            for r in range(num_restarts):
                if time.time() > deadline - 3.0:
                    break
                sub_budget = budget / (num_restarts - r)
                seed_adj = spectral_init(target_n, random)
                found = try_repair(target_n, sub_budget, seed_adj=seed_adj)
                if found is not None:
                    break
            
            if found is not None:
                best_n = target_n
                best_edges = found
                break  # Found a valid graph, try to go higher
    
    # Phase 2: If we found something, try to extend further
    if best_n > 17 and time.time() < deadline - 3.0:
        remaining = deadline - time.time() - 3.0
        for target in range(best_n + 1, 36):
            if time.time() > deadline - 2.0:
                break
            budget = min(remaining * 0.15, 5.0)
            found = None
            for r in range(3):
                if time.time() > deadline - 2.0:
                    break
                seed_adj = spectral_init(target, random)
                # Embed best graph in first best_n vertices
                seed_adj[:best_n, :best_n] = np.zeros((best_n, best_n), dtype=np.int8)
                for u, v in best_edges:
                    if u < best_n and v < best_n:
                        seed_adj[u, v] = 1
                        seed_adj[v, u] = 1
                found = try_repair(target, budget / (3 - r), seed_adj=seed_adj)
                if found is not None:
                    break
            
            if found is not None:
                best_n = target
                best_edges = found
            else:
                break
    
    # Phase 3: Fallback - incremental extension from 17 if nothing better found
    if best_n <= 17 and time.time() < deadline - 3.0:
        current_n = 17
        current_adj = adj17.copy()
        
        while current_n < 35 and time.time() < deadline - 3.0:
            remaining = deadline - time.time() - 3.0
            if remaining < 1.0:
                break
            
            target_n = current_n + 1
            budget = min(remaining * 0.2, 6.0)
            
            found = None
            for attempt in range(3):
                if time.time() > deadline - 3.0:
                    break
                seed_adj = np.zeros((target_n, target_n), dtype=np.int8)
                seed_adj[:current_n, :current_n] = current_adj
                p_connect = 0.4 + 0.1 * random.random()
                for j in range(current_n):
                    if random.random() < p_connect:
                        seed_adj[current_n, j] = 1
                        seed_adj[j, current_n] = 1
                found = try_repair(target_n, budget / (3 - attempt), seed_adj=seed_adj)
                if found is not None:
                    break
            
            if found is not None:
                current_n = target_n
                current_adj = np.zeros((current_n, current_n), dtype=np.int8)
                for u, v in found:
                    current_adj[u, v] = 1
                    current_adj[v, u] = 1
                best_n = current_n
                best_edges = found
            else:
                break
    
    return {'n': best_n, 'edges': best_edges}
