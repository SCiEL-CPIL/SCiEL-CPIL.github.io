import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

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

    def is_valid(n, adj):
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) > 0:
            return False
        if count_ind6_fast(n, masks) > 0:
            return False
        return True

    def edges_from_adj(n, adj):
        return [[i, j] for i in range(n) for j in range(i + 1, n) if adj[i, j]]

    def find_max_independent_set(n, adj, limit=6):
        """Find largest independent set up to limit using greedy + backtracking."""
        masks = build_masks(n, adj)
        full_mask = (1 << n) - 1
        comp_masks = [0] * n
        for u in range(n):
            comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
        
        best_size = 0
        best_set = []
        
        def backtrack(start, current, size):
            nonlocal best_size, best_set
            if size > best_size:
                best_size = size
                best_set = current[:]
                if size >= limit:
                    return
            # Upper bound: can't exceed remaining vertices
            if size + (n - start) <= best_size:
                return
            for v in range(start, n):
                # Check if v is independent from all in current
                valid = True
                for u in current:
                    if adj[u, v]:
                        valid = False
                        break
                if valid:
                    current.append(v)
                    backtrack(v + 1, current, size + 1)
                    current.pop()
        
        backtrack(0, [], 0)
        return best_size, best_set

    def greedy_construct(n, seed=None):
        """
        Greedy vertex-by-vertex construction.
        For each new vertex, choose edges to previous vertices that:
        1. Don't create a K4
        2. Help maintain alpha < 6
        """
        if seed is not None:
            random.seed(seed)
        
        adj = np.zeros((n, n), dtype=bool)
        
        for new_v in range(1, n):
            if time.time() > deadline:
                return None
            
            # Determine which previous vertices to connect to
            # Strategy: connect to vertices that are "safe" (won't create K4)
            # and prioritize connections that reduce independence
        
            candidate_edges = []
            for u in range(new_v):
                # Check if adding edge (new_v, u) would create a K4
                # A K4 would need: u, new_v, w, x where w,x are common neighbors of u and new_v
                # and w-x is an edge
                # Since new_v is new, its only potential common neighbors with u are
                # vertices connected to both u and new_v. But new_v has no edges yet.
                # So adding (new_v, u) can only create K4 if there exist w, x such that:
                #   - w and x are both neighbors of u
                #   - w-x is an edge
                #   - AND we also connect new_v to w and x (but we haven't yet)
                # Actually, adding a single edge (new_v, u) cannot create a K4 by itself
                # because new_v has degree 0 before this edge.
                # A K4 containing (new_v, u) needs 2 more vertices w, x that are
                # neighbors of both new_v and u, with w-x edge.
                # Since new_v only connects to u so far (in this iteration), 
                # no K4 is created by adding (new_v, u) alone.
                # But we need to be careful about the full set of edges we add.
                
                # For now, let's just track which edges we add and check at the end
                candidate_edges.append(u)
            
            # Randomly select a subset of candidate edges
            # Use a density that balances K4 avoidance and independence
            # Higher density -> fewer independent sets but more K4 risk
            # Lower density -> fewer K4 but more independent sets
            
            # Adaptive density: start moderate, adjust based on current state
            density = 0.45
            
            selected_edges = []
            for u in candidate_edges:
                if random.random() < density:
                    selected_edges.append(u)
            
            # Add selected edges
            for u in selected_edges:
                adj[new_v, u] = True
                adj[u, new_v] = True
            
            # Check if we've created a K4
            masks = build_masks(n, adj)
            if count_k4_fast(n, masks) > 0:
                # Need to remove some edges
                # Find and remove edges in K4s
                for _ in range(100):
                    if time.time() > deadline:
                        return None
                    
                    masks = build_masks(n, adj)
                    k4_count = count_k4_fast(n, masks)
                    if k4_count == 0:
                        break
                    
                    # Find an edge in a K4 and remove it
                    found = False
                    for u in range(new_v):
                        for v in range(u + 1, new_v + 1):
                            if adj[u, v]:
                                common = masks[u] & masks[v]
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
                                            # Found K4: u, v, wi, xi
                                            # Remove one of the edges
                                            adj[u, v] = False
                                            adj[v, u] = False
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                        if found:
                            break
                    
                    if not found:
                        break
            
            # Check independence
            masks = build_masks(n, adj)
            alpha, _ = find_max_independent_set(n, adj, limit=6)
            if alpha >= 6:
                # Need to add edges to break independent sets
                # Find the independent set and add edges within it
                _, ind_set = find_max_independent_set(n, adj, limit=6)
                if len(ind_set) >= 6:
                    # Add random edges within the independent set
                    for i in range(len(ind_set)):
                        for j in range(i + 1, len(ind_set)):
                            u, v = ind_set[i], ind_set[j]
                            if not adj[u, v] and random.random() < 0.3:
                                # Check if adding this edge creates a K4
                                adj[u, v] = True
                                adj[v, u] = True
                                masks = build_masks(n, adj)
                                if count_k4_fast(n, masks) > 0:
                                    # Revert
                                    adj[u, v] = False
                                    adj[v, u] = False
                                else:
                                    alpha, _ = find_max_independent_set(n, adj, limit=6)
                                    if alpha < 6:
                                        break
                        if alpha < 6:
                            break
        
        # Final validation
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) == 0 and count_ind6_fast(n, masks) == 0:
            return edges_from_adj(n, adj)
        
        return None

    def complement_repair(n, seed=None):
        """
        Start with complement of a random K4-free graph, then repair.
        The complement of a K4-free graph has no independent 4-set.
        We want the original to have no K4 and no independent 6-set.
        Equivalently, complement has no independent 4-set (i.e., no K4) 
        and clique number < 6.
        """
        if seed is not None:
            random.seed(seed)
        
        # Start with a random graph
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.5:
                    adj[i, j] = True
                    adj[j, i] = True
        
        # Phase 1: Remove edges to eliminate K4s
        for iteration in range(500):
            if time.time() > deadline:
                return None
            
            masks = build_masks(n, adj)
            k4_count = count_k4_fast(n, masks)
            
            if k4_count == 0:
                break
            
            # Find edge in most K4s and remove it
            best_edge = None
            best_participation = 0
            
            for u in range(n):
                for v in range(u + 1, n):
                    if adj[u, v]:
                        common = masks[u] & masks[v]
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
                        if part > best_participation:
                            best_participation = part
                            best_edge = (u, v)
            
            if best_edge is None:
                break
            
            u, v = best_edge
            adj[u, v] = False
            adj[v, u] = False
        
        # Check if K4-free
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) > 0:
            return None
        
        # Phase 2: Add edges to eliminate independent 6-sets
        for iteration in range(300):
            if time.time() > deadline:
                return None
            
            masks = build_masks(n, adj)
            ind6_count = count_ind6_fast(n, masks)
            
            if ind6_count == 0:
                break
            
            # Find an independent 6-set and add an edge within it
            # that doesn't create a K4
            full_mask = (1 << n) - 1
            comp_masks = [0] * n
            for u in range(n):
                comp_masks[u] = full_mask & ~(masks[u] | (1 << u))
            
            # Find a large independent set
            best_ind = []
            for start in range(n):
                current = [start]
                for v in range(start + 1, n):
                    valid = True
                    for u in current:
                        if adj[u, v]:
                            valid = False
                            break
                    if valid:
                        current.append(v)
                if len(current) > len(best_ind):
                    best_ind = current
                if len(best_ind) >= 6:
                    break
            
            if len(best_ind) < 6:
                break
            
            # Try to add edges within best_ind
            added = False
            for i in range(len(best_ind)):
                for j in range(i + 1, len(best_ind)):
                    u, v = best_ind[i], best_ind[j]
                    if adj[u, v]:
                        continue
                    
                    # Check if adding (u,v) creates a K4
                    masks = build_masks(n, adj)
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
                    added = True
                    break
                if added:
                    break
            
            if not added:
                break
        
        # Final check
        masks = build_masks(n, adj)
        if count_k4_fast(n, masks) == 0 and count_ind6_fast(n, masks) == 0:
            return edges_from_adj(n, adj)
        
        return None

    def simulated_annealing(n, max_steps=5000, seed=None):
        """
        Simulated annealing with edge flips.
        Objective: minimize K4_count + ind6_count.
        """
        if seed is not None:
            random.seed(seed)
        
        adj = np.zeros((n, n), dtype=bool)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.4:
                    adj[i, j] = True
                    adj[j, i] = True
        
        masks = build_masks(n, adj)
        current_cost = count_k4_fast(n, masks) + count_ind6_fast(n, masks)
        
        if current_cost == 0:
            return edges_from_adj(n, adj)
        
        T = 5.0
        best_cost = current_cost
        best_adj = adj.copy()
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random edge to flip
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            
            old = adj[u, v]
            adj[u, v] = not old
            adj[v, u] = not old
            
            masks = build_masks(n, adj)
            new_cost = count_k4_fast(n, masks) + count_ind6_fast(n, masks)
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_adj = adj.copy()
                if new_cost == 0:
                    return edges_from_adj(n, adj)
            
            if new_cost < current_cost:
                current_cost = new_cost
            else:
                delta = new_cost - current_cost
                T = max(0.1, T * 0.9995)
                if random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                else:
                    adj[u, v] = old
                    adj[v, u] = old
        
        if best_cost == 0:
            return edges_from_adj(n, best_adj)
        
        return None

    # Baseline: Clebsch graph (16 vertices)
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
    
    # Try to build larger graphs
    # Strategy: try complement_repair and simulated_annealing for increasing n
    
    for target_n in range(17, 36):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
        
        found = False
        
        # Try complement repair with multiple seeds
        for seed in range(5):
            if time.time() > deadline or found:
                break
            
            result = complement_repair(target_n, seed=seed)
            if result is not None:
                found = True
                best_n = target_n
                best_edges = result
                break
        
        if found:
            if best_n >= 35:
                break
            continue
        
        # Try simulated annealing
        for seed in range(3):
            if time.time() > deadline or found:
                break
            
            max_steps = min(5000, int(remaining * 2))
            result = simulated_annealing(target_n, max_steps=max_steps, seed=seed)
            if result is not None:
                found = True
                best_n = target_n
                best_edges = result
                break
        
        if found:
            if best_n >= 35:
                break
    
    # If we still have time, try greedy construction for even larger n
    if best_n < 35 and time.time() < deadline - 10:
        for target_n in range(best_n + 1, min(best_n + 8, 36)):
            if time.time() > deadline:
                break
            
            remaining = deadline - time.time()
            if remaining < 5.0:
                break
            
            found = False
            for seed in range(5):
                if time.time() > deadline or found:
                    break
                
                result = greedy_construct(target_n, seed=seed)
                if result is not None:
                    found = True
                    best_n = target_n
                    best_edges = result
                    break
            
            if found:
                if best_n >= 35:
                    break
    
    return {'n': best_n, 'edges': best_edges}
