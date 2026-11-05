import random
import time

def construct():
    n = 20
    start_time = time.time()
    max_time = 112.0  # Safety margin for 120s limit (with 8s buffer)
    
    def get_all_pairs(n):
        return [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    all_pairs = get_all_pairs(n)
    pair_to_idx = {}
    for idx, (u, v) in enumerate(all_pairs):
        pair_to_idx[(u, v)] = idx
        pair_to_idx[(v, u)] = idx
    
    m = len(all_pairs)  # 190
    
    # Build conflict graph: two edges conflict if they form a K2,3
    # Edges (a,b) and (c,d) form K2,3 if {a,b} and {c,d} are disjoint pairs
    # and they share exactly 3 common neighbors... 
    # Actually, K2,3 means two vertices with 3 common neighbors.
    # Two edges e1=(u1,v1) and e2=(u2,v2) conflict if there exist 3 vertices 
    # that are common neighbors of some pair... 
    # 
    # More precisely: a set of edges is K2,3-free iff no two vertices have 
    # 3 common neighbors. Two edges conflict in the MIS formulation if 
    # including both would force a K2,3.
    # 
    # Actually, the standard reduction: we want to select a maximum set of 
    # edges such that no two vertices have 3 common neighbors. This is NOT 
    # simply a maximum independent set on a pairwise conflict graph because 
    # the constraint is on triples of edges (or rather, on pairs of vertices 
    # having 3 common neighbors).
    #
    # Let me reconsider. The constraint is: for every pair (x,y), |N(x) ∩ N(y)| <= 2.
    # This is a 3-wise constraint on edges. So it's not a simple MIS problem.
    #
    # However, we can still use a branch-and-bound approach directly on the 
    # edge selection problem. Let's use a different B&B strategy.
    
    def is_valid_add(u, v, adj_mask, common):
        if (adj_mask[u] >> v) & 1:
            return False
        if common[u][v] > 2:
            return False
        mask_v = adj_mask[v]
        msk = mask_v
        while msk:
            lsb = msk & -msk
            w = lsb.bit_length() - 1
            if w != u and common[u][w] >= 2:
                return False
            msk -= lsb
        mask_u = adj_mask[u]
        msk = mask_u
        while msk:
            lsb = msk & -msk
            w = lsb.bit_length() - 1
            if w != v and common[v][w] >= 2:
                return False
            msk -= lsb
        return True

    def add_edge_to_state(u, v, adj_mask, common, edges):
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        edges.append([u, v])
        mask_v = adj_mask[v] & ~(1 << u)
        msk = mask_v
        while msk:
            lsb = msk & -msk
            w = lsb.bit_length() - 1
            common[u][w] += 1
            common[w][u] += 1
            msk -= lsb
        mask_u = adj_mask[u] & ~(1 << v)
        msk = mask_u
        while msk:
            lsb = msk & -msk
            w = lsb.bit_length() - 1
            common[v][w] += 1
            common[w][v] += 1
            msk -= lsb

    def remove_edge_from_state(u, v, adj_mask, common, edges):
        adj_mask[u] &= ~(1 << v)
        adj_mask[v] &= ~(1 << u)
        edges.remove([u, v])
        mask_v = adj_mask[v]
        msk = mask_v
        while msk:
            lsb = msk & -msk
            w = lsb.bit_length() - 1
            common[u][w] -= 1
            common[w][u] -= 1
            msk -= lsb
        mask_u = adj_mask[u]
        msk = mask_u
        while msk:
            lsb = msk & -msk
            w = lsb.bit_length() - 1
            common[v][w] -= 1
            common[w][v] -= 1
            msk -= lsb

    def greedy_fill(adj_mask, common, edges):
        pairs = all_pairs[:]
        random.shuffle(pairs)
        for u, v in pairs:
            if is_valid_add(u, v, adj_mask, common):
                add_edge_to_state(u, v, adj_mask, common, edges)

    def local_search_with_kicks(initial_edges, max_kicks=15):
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        for u, v in initial_edges:
            add_edge_to_state(u, v, adj_mask, common, edges)
        current_count = len(edges)
        best_local_edges = list(edges)
        best_local_count = current_count
        
        for kick_iter in range(max_kicks):
            # Try to add any valid edge repeatedly
            for step in range(100):
                if time.time() - start_time > max_time - 5:
                    return best_local_edges
                pairs = all_pairs[:]
                random.shuffle(pairs)
                added_any = False
                for u, v in pairs:
                    if is_valid_add(u, v, adj_mask, common):
                        add_edge_to_state(u, v, adj_mask, common, edges)
                        added_any = True
                        if len(edges) > best_local_count:
                            best_local_count = len(edges)
                            best_local_edges = list(edges)
                        break
                if not added_any:
                    break
            
            if not edges:
                break
            
            # Kick: remove 1-3 edges
            kick_size = random.randint(1, 3)
            to_remove = []
            available = edges[:]
            for _ in range(kick_size):
                if not available:
                    break
                idx = random.randint(0, len(available) - 1)
                to_remove.append(available[idx])
                available.pop(idx)
            for u, v in to_remove:
                remove_edge_from_state(u, v, adj_mask, common, edges)
            
            # Try to fill again
            for step in range(50):
                if time.time() - start_time > max_time - 5:
                    return best_local_edges
                pairs = all_pairs[:]
                random.shuffle(pairs)
                added = False
                for u, v in pairs:
                    if is_valid_add(u, v, adj_mask, common):
                        add_edge_to_state(u, v, adj_mask, common, edges)
                        added = True
                        if len(edges) > best_local_count:
                            best_local_count = len(edges)
                            best_local_edges = list(edges)
                        break
                if not added:
                    break
        
        return best_local_edges

    def branch_and_bound(initial_edges, time_limit):
        """
        B&B on the edge selection problem.
        State: (selected_edges, available_edges, adj_mask, common)
        We branch on whether to include or exclude each available edge.
        Upper bound: current_count + len(available_edges)
        """
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        selected = []
        for u, v in initial_edges:
            add_edge_to_state(u, v, adj_mask, common, selected)
        
        # Find all edges not yet selected
        selected_set = set()
        for u, v in selected:
            selected_set.add((u, v))
            selected_set.add((v, u))
        
        available = []
        for u, v in all_pairs:
            if (u, v) not in selected_set:
                available.append((u, v))
        
        best_count = len(selected)
        best_edges = list(selected)
        deadline = time.time() + time_limit
        
        # Sort available edges by some heuristic (e.g., degree potential)
        # For now, just use them in order
        
        def bb_recursive(adj_mask, common, selected, available, depth):
            nonlocal best_count, best_edges
            
            if time.time() > deadline:
                return
            
            if not available:
                if len(selected) > best_count:
                    best_count = len(selected)
                    best_edges = list(selected)
                return
            
            # Upper bound
            ub = len(selected) + len(available)
            if ub <= best_count:
                return
            
            # Try to add the first available edge
            u, v = available[0]
            rest = available[1:]
            
            # Branch 1: Include edge (u, v)
            if is_valid_add(u, v, adj_mask, common):
                # Save state
                adj_mask_save = list(adj_mask)
                common_save = [row[:] for row in common]
                selected_save = list(selected)
                
                add_edge_to_state(u, v, adj_mask, common, selected)
                bb_recursive(adj_mask, common, selected, rest, depth + 1)
                
                # Restore state
                adj_mask[:] = adj_mask_save
                for i in range(n):
                    common[i][:] = common_save[i]
                selected[:] = selected_save
            
            # Branch 2: Exclude edge (u, v)
            bb_recursive(adj_mask, common, selected, rest, depth + 1)
        
        # The full B&B is too slow for 190 edges. Let's use a different approach.
        # Instead, use a greedy + local search hybrid with better bounds.
        
        # For the B&B, let's work with a reduced problem:
        # Start from the best local search solution, and try to improve it
        # by exploring small perturbations systematically.
        
        return best_edges, best_count

    # Main loop
    best_edges = []
    best_count = 0
    
    # Phase 1: Run multiple local searches to find a good solution
    phase1_deadline = start_time + 60.0
    
    while time.time() < phase1_deadline:
        adj_mask = [0] * n
        common = [[0] * n for _ in range(n)]
        edges = []
        greedy_fill(adj_mask, common, edges)
        result_edges = local_search_with_kicks(edges)
        
        if len(result_edges) > best_count:
            best_count = len(result_edges)
            best_edges = result_edges
        
        # Check if we've likely found the optimum
        if best_count >= 150:  # Reasonable upper bound for K2,3-free on 20 vertices
            break
    
    # Phase 2: Try to improve with more sophisticated search
    if time.time() < max_time - 10:
        # Try different random seeds for diversity
        for seed in range(10):
            if time.time() > max_time - 10:
                break
            random.seed(seed * 12345 + 67890)
            adj_mask = [0] * n
            common = [[0] * n for _ in range(n)]
            edges = []
            greedy_fill(adj_mask, common, edges)
            result_edges = local_search_with_kicks(edges, max_kicks=20)
            
            if len(result_edges) > best_count:
                best_count = len(result_edges)
                best_edges = result_edges
    
    # Final Verification
    if not best_edges:
        best_edges = [[0, 1]]
    else:
        adj = [0] * n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        valid = True
        for u in range(n):
            for v in range(u + 1, n):
                if bin(adj[u] & adj[v]).count('1') > 2:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            return [[0, 1]]
    
    return best_edges

if __name__ == "__main__":
    edges = construct()
    print(f"Found {len(edges)} edges")
