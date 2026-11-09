import time
import random
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140  # Safety margin for 150s limit
    
    n = 16
    target_edges = 65
    all_edges = [(i,j) for i in range(n) for j in range(i+1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    best_score = float('inf')
    best_edge_set = None
    
    # Precompute neighbor lists for all possible edges? No, graph changes.
    # But we can precompute the list of all edges for quick random selection.
    
    def compute_score_and_penalty(adj_mask, deg):
        """
        Compute max triangle degree sum and sum of squared triangle degree sums.
        adj_mask: list of n integers, where adj_mask[u] is a bitmask of neighbors of u.
        deg: list of n integers, degrees.
        Returns: (max_sum, sum_sq)
        """
        max_sum = 0
        sum_sq = 0
        # Iterate over all edges
        for u in range(n):
            mask_u = adj_mask[u]
            # Iterate over neighbors v > u
            v = u + 1
            while v < n:
                if (mask_u >> v) & 1:
                    # Edge (u, v) exists
                    # Find common neighbors w > v
                    mask_v = adj_mask[v]
                    mask_u = adj_mask[u]
                    common = mask_u & mask_v
                    # We want w > v
                    common &= ~((1 << (v + 1)) - 1) # Mask out 0..v
                    # Iterate over set bits in common
                    w = v + 1
                    while w < n:
                        if (common >> w) & 1:
                            # Triangle (u, v, w)
                            s = deg[u] + deg[v] + deg[w]
                            if s > max_sum:
                                max_sum = s
                            sum_sq += s * s
                        w += 1
                v += 1
        return max_sum, sum_sq

    def get_adj_mask(deg, edge_set):
        adj_mask = [0] * n
        for u, v in edge_set:
            adj_mask[u] |= (1 << v)
            adj_mask[v] |= (1 << u)
        return adj_mask

    def random_init():
        """
        Generate a random graph with 65 edges.
        Strategy: Start with K_{8,8} (64 edges) and add 1 random edge.
        Then perform some random swaps to break symmetry.
        """
        edge_set = set()
        # K_{8,8}
        for u in range(8):
            for v in range(8, 16):
                edge_set.add((min(u,v), max(u,v)))
        
        # Add 1 edge
        # To minimize triangles, we might want an edge within a partition?
        # K_{8,8} is bipartite, so no triangles. Adding an edge within a part creates triangles.
        # Triangles will involve the new edge (u,v) and any common neighbor w.
        # If u,v in A, common neighbors are all in B (8 of them).
        # deg(u)=8+1=9, deg(v)=8+1=9, deg(w)=8. Sum = 26.
        # If we swap some edges, we might reduce degrees.
        
        # Random perturbation: swap 5 edges
        for _ in range(5):
            if len(edge_set) == 0: break
            e_remove = random.choice(list(edge_set))
            e_add = random.choice(all_edges)
            if e_add not in edge_set:
                edge_set.remove(e_remove)
                edge_set.add(e_add)
                
        # Ensure exactly 65 edges
        while len(edge_set) < target_edges:
            e = random.choice(all_edges)
            edge_set.add(e)
        while len(edge_set) > target_edges:
            e = random.choice(list(edge_set))
            edge_set.remove(e)
            
        return edge_set

    def local_search(edge_set_init, time_limit):
        """
        Local search using simulated annealing.
        Objective: Minimize max_triangle_sum.
        Secondary: Minimize sum_sq_triangle_sums (to flatten distribution).
        """
        nonlocal best_score, best_edge_set
        
        current_set = set(edge_set_init)
        current_deg = [0]*n
        for u, v in current_set:
            current_deg[u] += 1
            current_deg[v] += 1
        current_adj = get_adj_mask(current_deg, current_set)
        
        curr_max, curr_sq = compute_score_and_penalty(current_adj, current_deg)
        
        local_best_score = curr_max
        local_best_set = set(current_set)
        
        if curr_max < best_score:
            best_score = curr_max
            best_edge_set = set(current_set)
        
        temp = 10.0
        cooling = 0.999
        min_temp = 0.1
        
        while time.time() < time_limit:
            # Random move: remove one edge, add another
            e_remove = random.choice(list(current_set))
            e_add = random.choice(all_edges)
            
            if e_add in current_set:
                continue
            
            # Apply move
            current_set.remove(e_remove)
            current_set.add(e_add)
            
            # Update degrees
            u1, v1 = e_remove
            current_deg[u1] -= 1
            current_deg[v1] -= 1
            u2, v2 = e_add
            current_deg[u2] += 1
            current_deg[v2] += 1
            
            # Update adjacency mask
            # Remove e_remove
            current_adj[u1] &= ~(1 << v1)
            current_adj[v1] &= ~(1 << u1)
            # Add e_add
            current_adj[u2] |= (1 << v2)
            current_adj[v2] |= (1 << u2)
            
            new_max, new_sq = compute_score_and_penalty(current_adj, current_deg)
            
            # Acceptance criterion
            # We want to minimize max. 
            # If new_max < curr_max, accept.
            # If new_max == curr_max, prefer lower new_sq.
            # If new_max > curr_max, accept with probability based on delta.
            
            delta_max = new_max - curr_max
            delta_sq = new_sq - curr_sq
            
            accepted = False
            if delta_max < 0:
                accepted = True
            elif delta_max == 0:
                if delta_sq < 0:
                    accepted = True
                else:
                    # Accept if it helps flatten? Maybe.
                    if random.random() < math.exp(-delta_sq / (temp * 100)):
                        accepted = True
            else:
                # Penalize increase in max heavily
                # Use a surrogate: cost = delta_max * 100 + delta_sq * 0.1
                cost = delta_max * 100 + delta_sq * 0.01
                if random.random() < math.exp(-cost / temp):
                    accepted = True
            
            if accepted:
                curr_max = new_max
                curr_sq = new_sq
                if curr_max < local_best_score:
                    local_best_score = curr_max
                    local_best_set = set(current_set)
                    if curr_max < best_score:
                        best_score = curr_max
                        best_edge_set = set(current_set)
            else:
                # Revert
                current_set.remove(e_add)
                current_set.add(e_remove)
                
                current_deg[u2] -= 1
                current_deg[v2] -= 1
                current_deg[u1] += 1
                current_deg[v1] += 1
                
                current_adj[u2] &= ~(1 << v2)
                current_adj[v2] &= ~(1 << u2)
                current_adj[u1] |= (1 << v1)
                current_adj[v1] |= (1 << u1)
            
            temp = max(temp * cooling, min_temp)
            
            # Occasional 2-swap for better mixing
            if random.random() < 0.1:
                e_rem1 = random.choice(list(current_set))
                e_rem2 = random.choice(list(current_set))
                if e_rem1 == e_rem2:
                    continue
                e_add1 = random.choice(all_edges)
                e_add2 = random.choice(all_edges)
                if e_add1 in current_set or e_add2 in current_set or e_add1 == e_add2:
                    continue
                if e_add1 == e_rem1 or e_add1 == e_rem2 or e_add2 == e_rem1 or e_add2 == e_rem2:
                    continue
                
                old_set = set(current_set)
                old_deg = list(current_deg)
                old_adj = list(current_adj)
                
                current_set.remove(e_rem1)
                current_set.remove(e_rem2)
                current_set.add(e_add1)
                current_set.add(e_add2)
                
                # Update deg
                u1, v1 = e_rem1
                current_deg[u1] -= 1; current_deg[v1] -= 1
                u2, v2 = e_rem2
                current_deg[u2] -= 1; current_deg[v2] -= 1
                u3, v3 = e_add1
                current_deg[u3] += 1; current_deg[v3] += 1
                u4, v4 = e_add2
                current_deg[u4] += 1; current_deg[v4] += 1
                
                # Update adj
                current_adj[u1] &= ~(1 << v1); current_adj[v1] &= ~(1 << u1)
                current_adj[u2] &= ~(1 << v2); current_adj[v2] &= ~(1 << u2)
                current_adj[u3] |= (1 << v3); current_adj[v3] |= (1 << u4) # Wait, u4 is v of add2? No.
                current_adj[u3] |= (1 << v3); current_adj[v3] |= (1 << u3)
                current_adj[u4] |= (1 << v4); current_adj[v4] |= (1 << u4)
                
                new_max, new_sq = compute_score_and_penalty(current_adj, current_deg)
                
                delta_max = new_max - curr_max
                delta_sq = new_sq - curr_sq
                
                accepted = False
                if delta_max < 0:
                    accepted = True
                elif delta_max == 0:
                    if delta_sq < 0:
                        accepted = True
                else:
                    cost = delta_max * 100 + delta_sq * 0.01
                    if random.random() < math.exp(-cost / temp):
                        accepted = True
                
                if accepted:
                    curr_max = new_max
                    curr_sq = new_sq
                    if curr_max < local_best_score:
                        local_best_score = curr_max
                        local_best_set = set(current_set)
                        if curr_max < best_score:
                            best_score = curr_max
                            best_edge_set = set(current_set)
                else:
                    # Revert
                    current_set.clear()
                    current_set.update(old_set)
                    current_deg = old_deg
                    current_adj = old_adj

        return local_best_score, local_best_set

    # Main loop
    num_restarts = 100
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Initialize
        es_init = random_init()
        
        # Run local search
        time_left = deadline - 5 - time.time()
        if time_left < 1:
            break
        
        # Allocate time per restart
        # If we have many restarts, give less time to each.
        # If we have few, give more.
        search_time = time.time() + max(1.0, min(10.0, time_left / (num_restarts - i) * 0.8))
        
        final_score, final_set = local_search(es_init, search_time)
        
        if final_score < best_score:
            best_score = final_score
            best_edge_set = final_set

    if best_edge_set is None:
        # Fallback: K_{8,8} + (0,1)
        best_edge_set = set()
        for u in range(8):
            for v in range(8, 16):
                best_edge_set.add((u, v))
        best_edge_set.add((0, 1))
        
    edge_list = [[u, v] for (u, v) in best_edge_set]
    return {'n': n, 'edges': edge_list}
