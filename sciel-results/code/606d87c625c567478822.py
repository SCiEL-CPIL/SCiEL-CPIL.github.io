import time
import numpy as np
from itertools import combinations

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin
    
    # Precompute all triples for cleaning phase
    triples = list(combinations(range(n), 3))
    
    # Precompute all pairs
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def can_add_mask(adj_masks, u, v):
        """
        Check if adding edge (u,v) violates K3,3.
        Optimized version using neighbor lists.
        """
        # Case 1: u is in Part A, v is in Part B.
        # Part A = {u, a, b}, Part B = {v, c, d}.
        # Violation if v is common neighbor of a, b AND |N(u) \cap N(a) \cap N(b)| >= 2.
        # This implies v is a NEW common neighbor for {u, a, b}.
        # So we need v in N(a) and v in N(b).
        
        # Get neighbors of v (excluding u, since u-v is not an edge yet)
        # We only care about a, b that are neighbors of v.
        neighbors_v = []
        mask_v = adj_masks[v]
        for x in range(n):
            if x != u and (mask_v >> x) & 1:
                neighbors_v.append(x)
                
        for i_a in range(len(neighbors_v)):
            a = neighbors_v[i_a]
            ma = adj_masks[a]
            for i_b in range(i_a + 1, len(neighbors_v)):
                b = neighbors_v[i_b]
                mb = adj_masks[b]
                # Check if |N(u) \cap N(a) \cap N(b)| >= 2
                # Note: v is not in N(u) yet, so v is not in this intersection.
                common = adj_masks[u] & ma & mb
                if common.bit_count() >= 2:
                    return False
        
        # Case 2: v is in Part A, u is in Part B.
        # Part A = {v, a, b}, Part B = {u, c, d}.
        # Violation if u is common neighbor of a, b AND |N(v) \cap N(a) \cap N(b)| >= 2.
        # So we need u in N(a) and u in N(b).
        
        neighbors_u = []
        mask_u = adj_masks[u]
        for x in range(n):
            if x != v and (mask_u >> x) & 1:
                neighbors_u.append(x)
                
        for i_a in range(len(neighbors_u)):
            a = neighbors_u[i_a]
            ma = adj_masks[a]
            for i_b in range(i_a + 1, len(neighbors_u)):
                b = neighbors_u[i_b]
                mb = adj_masks[b]
                # Check if |N(v) \cap N(a) \cap N(b)| >= 2
                common = adj_masks[v] & ma & mb
                if common.bit_count() >= 2:
                    return False
        
        return True

    best_edges = []
    best_count = 0
    
    rng = np.random.default_rng(42)
    
    # Strategy: 
    # 1. Start with random graph
    # 2. Remove edges to make K3,3 free
    # 3. Greedy add
    # 4. Local search
    
    num_restarts = 500 # Increased due to speedup
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        p = 0.4 + 0.3 * rng.random()
        adj_masks = [0] * n
        
        for i in range(n):
            for j in range(i+1, n):
                if rng.random() < p:
                    adj_masks[i] |= (1 << j)
                    adj_masks[j] |= (1 << i)
                    
        # Clean - remove violating edges
        max_clean_iters = 500
        for _ in range(max_clean_iters):
            if time.time() > deadline - 5:
                break
            found_violation = False
            # Randomly pick a triple to check to speed up? 
            # Or iterate all. With n=20, 1140 triples is small.
            for i, j, k in triples:
                common = adj_masks[i] & adj_masks[j] & adj_masks[k]
                if common.bit_count() >= 3:
                    # Find neighbors in common
                    neighbors = []
                    for x in range(n):
                        if (common >> x) & 1:
                            neighbors.append(x)
                    # Pick random edges from the K3,3 to remove
                    removed = False
                    for _try in range(20):
                        vi = [i, j, k][rng.integers(0, 3)]
                        vx = neighbors[rng.integers(0, len(neighbors))]
                        if (adj_masks[vi] >> vx) & 1:
                            adj_masks[vi] &= ~(1 << vx)
                            adj_masks[vx] &= ~(1 << vi)
                            removed = True
                            break
                    if not removed:
                        # Fallback
                        vi, vx = i, neighbors[0]
                        adj_masks[vi] &= ~(1 << vx)
                        adj_masks[vx] &= ~(1 << vi)
                    found_violation = True
                    break
            if not found_violation:
                break
        
        # Rebuild edges
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if (adj_masks[i] >> j) & 1:
                    edges.append((i, j))
        
        current_count = len(edges)
        
        # Greedy Add
        # Use a list of missing edges
        missing = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_masks[i] >> j) & 1)]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > deadline - 2:
                break
            if can_add_mask(adj_masks, u, v):
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append((u, v))
                current_count += 1
        
        # Local Search
        # We will try to remove one edge and add up to 2.
        # To speed up, we maintain a list of missing edges.
        # But for simplicity and robustness, we rebuild or update carefully.
        
        improved = True
        max_local_iters = 100 # Increased iterations
        
        for _ in range(max_local_iters):
            if time.time() > deadline - 5:
                break
            improved = False
            edge_list = list(edges)
            rng.shuffle(edge_list)
            
            for u, v in edge_list:
                if time.time() > deadline - 5:
                    break
                
                # Remove (u,v)
                adj_masks[u] &= ~(1 << v)
                adj_masks[v] &= ~(1 << u)
                
                # Update edges list? 
                # It's safer to just track the set of edges in a set for O(1) lookup/removal?
                # But `edges` is a list. `edges.remove` is O(N).
                # Let's use a set for active edges for faster removal/addition in local search?
                # The current code uses a list.
                
                # Let's stick to the list but be careful.
                try:
                    edges.remove((u, v))
                except ValueError:
                    # Should not happen
                    pass
                
                # Try to add up to 2 edges
                # Get missing edges
                # Optimization: Instead of rebuilding all missing, 
                # we can just check the ones that are "likely" or just shuffle the previous missing list?
                # The previous missing list might have changed.
                # Rebuilding is O(N^2). N=20, N^2=400. 400 ops is fast.
                missing_now = [(i, j) for i in range(n) for j in range(i+1, n) if not ((adj_masks[i] >> j) & 1)]
                rng.shuffle(missing_now)
                
                added = 0
                for a, b in missing_now:
                    if added >= 2:
                        break
                    if can_add_mask(adj_masks, a, b):
                        adj_masks[a] |= (1 << b)
                        adj_masks[b] |= (1 << a)
                        edges.append((a, b))
                        added += 1
                        
                if added > 1:
                    improved = True
                elif added == 0:
                    # Restore
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.append((u, v))
                else:
                    # added == 1, no net gain, restore to keep state simple? 
                    # Or keep the swap? 
                    # If we swap 1 for 1, count is same. 
                    # Might help future swaps.
                    # Let's keep the swap (do not restore) to allow exploration.
                    # But `improved` remains False, so we might stop if no 2-for-1 found.
                    # To allow 1-for-1 moves, we should mark improved if added==1?
                    # No, usually we want to increase count.
                    # If we do 1-for-1, we might get stuck.
                    # Let's restore if added <= 1 to ensure we only accept improvements?
                    # Or accept 1-for-1 to move in the search space?
                    # Let's accept 1-for-1 with low probability?
                    # For now, restore if added < 2.
                    adj_masks[u] |= (1 << v)
                    adj_masks[v] |= (1 << u)
                    edges.append((u, v))
                    
            if not improved:
                break
        
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]
            
            if best_count >= 60:
                if time.time() > deadline - 20:
                    break

    return [list(e) for e in best_edges]
