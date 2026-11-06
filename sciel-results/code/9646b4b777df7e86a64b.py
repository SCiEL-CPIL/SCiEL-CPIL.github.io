"""Spectral optimization approach for K4-free graphs on 18 vertices."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s kill
    n = 18
    random.seed(42)
    np.random.seed(42)
    
    def is_k4_free(masks):
        # Check no K4 using bitmasks
        for a in range(n):
            for b in range(a+1, n):
                if not (masks[a] >> b) & 1:
                    continue
                common = masks[a] & masks[b]
                common &= ~(1 << a)
                common &= ~(1 << b)
                c_bits = common
                while c_bits:
                    c = c_bits & (-c_bits)
                    c_idx = c.bit_length() - 1
                    c_bits ^= c
                    if (masks[c_idx] & common & ~(c - 1) & ~c) != 0:
                        return False
        return True
    
    def max_triangle_free_induced(masks):
        # Find largest subset of vertices inducing a triangle-free graph
        best = 0
        all_vertices = (1 << n) - 1
        
        def can_add(current_mask, v):
            v_neigh_in_current = masks[v] & current_mask
            bits = v_neigh_in_current
            while bits:
                b = bits & (-bits)
                u = b.bit_length() - 1
                bits ^= b
                if (masks[u] & v_neigh_in_current & ~b) != 0:
                    return False
            return True
        
        def search(current_mask, candidates_mask, depth):
            nonlocal best
            if time.time() > deadline - 5:
                return
            if current_mask.bit_count() + candidates_mask.bit_count() <= best:
                return
            if candidates_mask == 0:
                if current_mask.bit_count() > best:
                    best = current_mask.bit_count()
                return
            
            v_bit = candidates_mask & (-candidates_mask)
            v = v_bit.bit_length() - 1
            remaining = candidates_mask ^ v_bit
            
            if can_add(current_mask, v):
                new_current = current_mask | v_bit
                new_cands = 0
                temp = remaining
                while temp:
                    b = temp & (-temp)
                    idx = b.bit_length() - 1
                    temp ^= b
                    if can_add(new_current, idx):
                        new_cands |= b
                search(new_current, new_cands, depth + 1)
            
            search(current_mask, remaining, depth + 1)
        
        search(0, all_vertices, 0)
        return best
    
    def spectral_optimize():
        """
        Use spectral methods to find a good initial K4-free graph.
        We want to minimize the largest triangle-free induced subgraph.
        
        Strategy: Start from a random graph, then use gradient-like updates
        to push edges toward configurations that break up large triangle-free sets.
        """
        # Initialize with a random graph
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.5:
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0
        
        # Convert to bitmask representation for checking
        def adj_to_mask(adj_mat):
            masks = [0] * n
            for i in range(n):
                for j in range(n):
                    if adj_mat[i][j] > 0.5:
                        masks[i] |= (1 << j)
            return masks
        
        # Objective: we want to minimize max triangle-free induced subgraph size
        # But this is discrete. Instead, use a proxy:
        # Minimize the number of "near-triangle" patterns that could form large triangle-free sets
        
        # Alternative approach: Use simulated annealing on the edge set
        best_adj = adj.copy()
        best_score = float('inf')
        
        current_masks = adj_to_mask(adj)
        if not is_k4_free(current_masks):
            # Fix K4 violations by removing edges
            for a in range(n):
                for b in range(a+1, n):
                    if (current_masks[a] >> b) & 1:
                        common = current_masks[a] & current_masks[b]
                        common &= ~(1 << a)
                        common &= ~(1 << b)
                        c_bits = common
                        while c_bits:
                            c = c_bits & (-c_bits)
                            c_idx = c.bit_length() - 1
                            c_bits ^= c
                            d_bits = common & ~(c - 1) & ~c
                            while d_bits:
                                d = d_bits & (-d_bits)
                                d_idx = d.bit_length() - 1
                                d_bits ^= d
                                if (current_masks[c_idx] >> d_idx) & 1:
                                    # Found K4: a,b,c,d - remove edge (a,b)
                                    adj[a][b] = 0.0
                                    adj[b][a] = 0.0
                                    break
                            if adj[a][b] == 0.0:
                                break
                        if adj[a][b] == 0.0:
                            break
            current_masks = adj_to_mask(adj)
        
        current_score = max_triangle_free_induced(current_masks)
        if current_score < best_score:
            best_score = current_score
            best_adj = adj.copy()
        
        # Simulated annealing
        temp = 1.0
        cooling = 0.995
        iterations = 0
        
        while time.time() < deadline - 30 and temp > 0.01:
            # Pick a random edge to flip
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u == v:
                continue
            if u > v:
                u, v = v, u
            
            # Try flipping edge (u,v)
            adj[u][v] = 1.0 - adj[u][v]
            adj[v][u] = adj[u][v]
            
            new_masks = adj_to_mask(adj)
            
            # Check K4-free
            if is_k4_free(new_masks):
                new_score = max_triangle_free_induced(new_masks)
                
                # Accept if better, or with probability exp(-delta/temp)
                delta = new_score - current_score
                if delta <= 0 or random.random() < np.exp(-delta / temp):
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_adj = adj.copy()
                else:
                    # Reject: flip back
                    adj[u][v] = 1.0 - adj[u][v]
                    adj[v][u] = adj[u][v]
            else:
                # Reject K4 violation: flip back
                adj[u][v] = 1.0 - adj[u][v]
                adj[v][u] = adj[u][v]
            
            temp *= cooling
            iterations += 1
        
        # Convert best_adj to bitmask
        best_masks = adj_to_mask(best_adj)
        return best_masks, best_score
    
    def local_search(masks, current_best):
        """Local search: try edge additions and removals to improve."""
        all_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                all_edges.add((u, v))
        
        current_edges = set()
        for u in range(n):
            for v in range(u+1, n):
                if (masks[u] >> v) & 1:
                    current_edges.add((u, v))
        
        no_improve_count = 0
        max_no_improve = 500
        
        while time.time() < deadline - 15 and no_improve_count < max_no_improve:
            # Try random edge modifications
            missing = all_edges - current_edges
            improved = False
            
            if missing:
                candidate_edge = random.choice(list(missing))
                u, v = candidate_edge
                new_mask = masks[:]
                new_mask[u] |= (1 << v)
                new_mask[v] |= (1 << u)
                if is_k4_free(new_mask):
                    score = max_triangle_free_induced(new_mask)
                    if score < current_best:
                        masks = new_mask
                        current_edges.add(candidate_edge)
                        current_best = score
                        improved = True
                        no_improve_count = 0
                        continue
            
            if current_edges:
                candidate_edge = random.choice(list(current_edges))
                u, v = candidate_edge
                new_mask = masks[:]
                new_mask[u] &= ~(1 << v)
                new_mask[v] &= ~(1 << u)
                score = max_triangle_free_induced(new_mask)
                if score < current_best:
                    masks = new_mask
                    current_edges.discard(candidate_edge)
                    current_best = score
                    improved = True
                    no_improve_count = 0
                    continue
            
            if not improved:
                no_improve_count += 1
        
        return masks, current_best
    
    # Phase 1: Spectral/simulated annealing initialization
    init_masks, init_score = spectral_optimize()
    
    # Phase 2: Local search refinement
    best_masks, best_score = local_search(init_masks, init_score)
    
    # Phase 3: Try multiple random restarts if time permits
    restarts = 0
    while time.time() < deadline - 20 and restarts < 3:
        # New random initialization
        adj = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < 0.4:  # Sparser initial graph
                    adj[i][j] = 1.0
                    adj[j][i] = 1.0
        
        def adj_to_mask_local(adj_mat):
            masks_local = [0] * n
            for i in range(n):
                for j in range(n):
                    if adj_mat[i][j] > 0.5:
                        masks_local[i] |= (1 << j)
            return masks_local
        
        current_masks_local = adj_to_mask_local(adj)
        
        # Fix K4 violations
        for a in range(n):
            for b in range(a+1, n):
                if (current_masks_local[a] >> b) & 1:
                    common = current_masks_local[a] & current_masks_local[b]
                    common &= ~(1 << a)
                    common &= ~(1 << b)
                    c_bits = common
                    while c_bits:
                        c = c_bits & (-c_bits)
                        c_idx = c.bit_length() - 1
                        c_bits ^= c
                        d_bits = common & ~(c - 1) & ~c
                        while d_bits:
                            d = d_bits & (-d_bits)
                            d_idx = d.bit_length() - 1
                            d_bits ^= d
                            if (current_masks_local[c_idx] >> d_idx) & 1:
                                adj[a][b] = 0.0
                                adj[b][a] = 0.0
                                break
                        if adj[a][b] == 0.0:
                            break
                    if adj[a][b] == 0.0:
                        break
        current_masks_local = adj_to_mask_local(adj)
        
        if not is_k4_free(current_masks_local):
            continue
        
        current_score_local = max_triangle_free_induced(current_masks_local)
        
        # Brief local search
        if time.time() < deadline - 25:
            current_masks_local, current_score_local = local_search(current_masks_local, current_score_local)
        
        if current_score_local < best_score:
            best_masks = current_masks_local
            best_score = current_score_local
        
        restarts += 1
    
    # Convert best_masks to edge list
    result = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_masks[u] >> v) & 1:
                result.append([u, v])
    return result
# EVOLVE-BLOCK-END
