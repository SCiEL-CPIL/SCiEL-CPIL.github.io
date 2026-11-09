import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def build_masks(n, adj):
        masks = [0] * n
        for u in range(n):
            m = 0
            for v in range(n):
                if adj[u, v]:
                    m |= (1 << v)
            masks[u] = m
        return masks

    def count_k4_fast(n, masks):
        """Count K4s using bit operations."""
        k4_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (masks[u] & (1 << v)):
                    continue
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
        return k4_count

    def count_ind6_fast(n, masks):
        """Count independent 6-sets using bit operations."""
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        ind6_count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if not (comp_masks[u] & (1 << v)):
                    continue
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
                            y_mask = common2
                            while y_mask:
                                lsb_y = y_mask & -y_mask
                                yi = lsb_y.bit_length() - 1
                                y_mask ^= lsb_y
                                z_mask = common2 & ~((1 << (yi + 1)) - 1)
                                while z_mask:
                                    lsb_z = z_mask & -z_mask
                                    zi = lsb_z.bit_length() - 1
                                    z_mask ^= lsb_z
                                    if comp_masks[yi] & (1 << zi):
                                        ind6_count += 1
        return ind6_count

    def total_violations(n, adj):
        masks = build_masks(n, adj)
        return count_k4_fast(n, masks) + count_ind6_fast(n, masks)

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) > 0:
            return False
        if count_ind6_fast(n, masks) > 0:
            return False
        return True

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    # Clebsch graph: 16 vertices, K4-free, alpha=5. Valid baseline.
    clebsch_edges = [
        (0, 6), (0, 7), (0, 10), (0, 11),
        (1, 5), (1, 8), (1, 10), (1, 13),
        (2, 4), (2, 9), (2, 11), (2, 13),
        (3, 4), (3, 7), (3, 9), (3, 12),
        (5, 6), (5, 8), (5, 12), (5, 15),
        (6, 7), (6, 9), (6, 14), (6, 15),
        (7, 8), (7, 11), (7, 14), (7, 15),
        (8, 9), (8, 12), (8, 15),
        (9, 10), (9, 14),
        (10, 11), (10, 13), (10, 14),
        (11, 12), (11, 14), (11, 15),
        (12, 13), (12, 15),
        (13, 14), (13, 15)
    ]
    
    best_n = 16
    best_edges = clebsch_edges

    # Verify baseline
    adj_16 = np.zeros((16, 16), dtype=bool)
    for u, v in clebsch_edges:
        adj_16[u, v] = True
        adj_16[v, u] = True
    
    if not is_valid(16, adj_16):
        best_n = 5
        best_edges = []
    
    # NEW STRATEGY: Greedy edge-removal repair
    # Start with dense random graphs, then iteratively remove edges 
    # that participate in the most K4s while checking ind6 count.
    
    def greedy_repair(n, density, max_iters=2000):
        """
        Start with a random graph at given density, then greedily 
        remove edges that are in the most K4s until K4-free.
        Then check if ind6-free. If not, try adding edges back 
        where it doesn't create K4s.
        """
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < density:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Phase 1: Remove edges to eliminate K4s
        for it in range(max_iters):
            if time.time() > deadline:
                return None
            
            masks = build_masks(n, adj)
            k4_count = count_k4_fast(n, masks)
            
            if k4_count == 0:
                break
            
            # Find edge in most K4s
            # For each edge (u,v), count how many K4s it's in
            # A K4 containing (u,v) needs two common neighbors w,x with edge (w,x)
            edge_k4_participation = {}
            
            # Sample some edges to find high-participation ones
            candidates = []
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        common = masks[u] & masks[v]
                        # Count pairs (w,x) in common where w-x is an edge
                        part = 0
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
                                if adj[wi, xi]:
                                    part += 1
                        if part > 0:
                            candidates.append((part, u, v))
            
            if not candidates:
                break
            
            # Remove edge with highest participation
            candidates.sort(reverse=True)
            part, u, v = candidates[0]
            adj[u, v] = False
            adj[v, u] = False
        
        # Check if valid
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) == 0 and count_ind6_fast(n, masks) == 0:
            return edges_from_adj(n, adj)
        
        # Phase 2: Try to add edges back where it doesn't create K4
        # but first we need to be K4-free
        if count_k4_fast(n, masks) > 0:
            return None
        
        # Try adding edges to reduce ind6
        ind6_count = count_ind6_fast(n, masks)
        if ind6_count == 0:
            return edges_from_adj(n, adj)
        
        # Try adding random edges that don't create K4
        for attempt in range(500):
            if time.time() > deadline:
                break
            
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or adj[u, v]:
                continue
            
            # Check if adding (u,v) creates a K4
            common = masks[u] & masks[v]
            creates_k4 = False
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
                    if adj[wi, xi]:
                        creates_k4 = True
                        break
                if creates_k4:
                    break
            
            if creates_k4:
                continue
            
            # Add the edge
            adj[u, v] = True
            adj[v, u] = True
            masks[u] |= (1 << v)
            masks[v] |= (1 << u)
            
            # Check if still valid
            if count_k4_fast(n, masks) == 0 and count_ind6_fast(n, masks) == 0:
                return edges_from_adj(n, adj)
            
            # Revert
            adj[u, v] = False
            adj[v, u] = False
            masks[u] &= ~(1 << v)
            masks[v] &= ~(1 << u)
        
        return None

    # Try different target sizes with greedy repair
    for target_n in range(17, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        found = False
        
        # Try different densities
        densities = [0.3, 0.4, 0.5, 0.35, 0.45]
        random.shuffle(densities)
        
        for density in densities:
            if time.time() > deadline or found:
                break
            
            result = greedy_repair(target_n, density)
            if result is not None:
                found = True
                best_n = target_n
                best_edges = result
                break
        
        if found and best_n >= 35:
            break
    
    # If we didn't find anything better, also try simulated annealing as backup
    if best_n < 30 and time.time() < deadline - 10:
        for target_n in range(best_n + 1, min(best_n + 5, 36)):
            if time.time() > deadline:
                break
            
            adj = np.zeros((target_n, target_n), dtype=bool)
            for i in range(target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.4:
                        adj[i, j] = True
                        adj[j, i] = True
            
            current_cost = total_violations(target_n, adj)
            
            if current_cost == 0:
                best_n = target_n
                best_edges = edges_from_adj(target_n, adj)
                break
            
            T = 1.0
            no_improve = 0
            max_steps = 200
            
            for step in range(max_steps):
                if time.time() > deadline:
                    break
                if no_improve > 100:
                    break
                
                u = random.randint(0, target_n - 1)
                v = random.randint(0, target_n - 1)
                if u == v:
                    continue
                
                old = adj[u, v]
                adj[u, v] = not old
                adj[v, u] = not old
                
                new_cost = total_violations(target_n, adj)
                
                if new_cost < current_cost:
                    current_cost = new_cost
                    no_improve = 0
                    if new_cost == 0:
                        best_n = target_n
                        best_edges = edges_from_adj(target_n, adj)
                        break
                else:
                    delta = new_cost - current_cost
                    T = max(0.1, T * 0.998)
                    if random.random() < np.exp(-delta / T):
                        current_cost = new_cost
                    else:
                        adj[u, v] = old
                        adj[v, u] = old
                    no_improve += 1
    
    return {'n': best_n, 'edges': best_edges}
