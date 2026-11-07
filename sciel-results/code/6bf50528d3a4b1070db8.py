"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 15
    # All possible edges in a simple undirected graph on n vertices
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)
    
    # Pre-compute edge indices for fast lookup
    edge_to_idx = {e: i for i, e in enumerate(all_edges)}
    idx_to_edge = {i: e for i, e in enumerate(all_edges)}

    # Bitmasks for adjacency and degrees
    # deg_mask[d] has bit i set if deg[i] == d
    # adj_mask[i] has bit j set if (i, j) is an edge
    
    def check_valid(adj_masks, deg_masks):
        """
        Returns True if the graph defined by adj_masks and deg_masks is valid.
        Validity condition: No simple path s-u-v-t with deg(s) == deg(t).
        """
        # For each edge (u, v), we need to ensure:
        # For all s in N(u)\{v} and t in N(v)\{u}, s != t, deg(s) != deg(t).
        
        # Let S_degs = {deg(s) | s in N(u)\{v}}
        # Let T_degs = {deg(t) | t in N(v)\{u}}
        # We need S_degs ∩ T_degs = ∅.
        
        # However, we must also handle the s == t case?
        # "simple three-edge path s-u-v-t" implies s, u, v, t are distinct?
        # Usually simple path means vertices are distinct.
        # The prompt says "simple three-edge path". This implies s, u, v, t are distinct.
        # So s != t is required.
        
        # Let's verify: "Reject if any simple three-edge path has endpoints of equal degree"
        # A path of length 3 is s-u-v-t.
        # Simple means s != u != v != t and s != v and s != t and u != t.
        # Since it's a path, edges are (s,u), (u,v), (v,t).
        # So s and t are endpoints.
        
        # If S_degs and T_degs intersect at degree d, it means there exists s with deg d and t with deg d.
        # If s != t, it's a violation.
        # If s == t, it's not a simple path (cycle of length 3? No, s-u-v-s is a triangle, length 3 edges but vertices s,u,v. Wait, s-u-v-t with s=t is a cycle. A simple path usually doesn't close. Even if it did, the endpoints would be the same vertex, so degrees are trivially equal. But is s-u-v-s a "simple path"? No, it's a cycle. A simple path has distinct vertices. So s != t.)
        
        # So the condition is:
        # For each edge (u, v):
        #   Let A = {deg(s) for s in N(u) if s != v}
        #   Let B = {deg(t) for t in N(v) if t != u}
        #   If A ∩ B is not empty, we must check if there is a pair (s, t) with s != t.
        #   If the only s in N(u)\{v} with deg d is the same vertex as the only t in N(v)\{u} with deg d, then no violation for that degree d?
        #   Actually, if s == t, then s-u-v-s is a cycle. Is it a "simple path"?
        #   Standard definition: A path is a sequence of distinct vertices. So s, u, v, t must be distinct.
        #   Therefore, if s == t, it is NOT a valid path instance.
        
        # So, for a specific degree d in A ∩ B:
        #   Let S_d = {s in N(u)\{v} | deg(s) == d}
        #   Let T_d = {t in N(v)\{u} | deg(t) == d}
        #   Violation exists if there is s in S_d and t in T_d such that s != t.
        #   This is true if:
        #     |S_d| > 1 (then any s1, s2 works? No, we need s in S_d, t in T_d. If |S_d|>=1 and |T_d|>=1.
        #     If S_d == T_d and |S_d| == 1, then s==t is the only option. No violation.
        #     Otherwise, violation.
        
        # Efficient check using bitmasks:
        # For each u, v where adj[u] has bit v:
        #   mask_u_ex_v = adj_masks[u] & ~(1 << v)
        #   mask_v_ex_u = adj_masks[v] & ~(1 << u)
        
        #   We need to check if there is a degree d such that:
        #   (mask_u_ex_v & deg_masks[d]) and (mask_v_ex_u & deg_masks[d]) are non-empty,
        #   AND they don't share exactly one common vertex which is the only one in both sets.
        
        #   Let M_u_d = mask_u_ex_v & deg_masks[d]
        #   Let M_v_d = mask_v_ex_u & deg_masks[d]
        
        #   If M_u_d == 0 or M_v_d == 0: continue.
        #   Common = M_u_d & M_v_d.
        #   If Common == M_u_d == M_v_d and bit_count(Common) == 1:
        #       # Only one vertex s=t. No violation for this d.
        #       pass
        #   Else:
        #       # There is s != t. Violation.
        #       return False
        
        # This requires iterating over d (0..14).
        # 15 edges * 15 degrees * bit ops. Very fast.
        
        for u in range(n):
            mask_u = adj_masks[u]
            # Iterate neighbors v
            v = 0
            while mask_u:
                if mask_u & 1:
                    # v is a neighbor
                    mask_v = adj_masks[v]
                    
                    mask_u_ex_v = mask_u & ~(1 << v)
                    mask_v_ex_u = mask_v & ~(1 << u)
                    
                    # Check each degree
                    valid_deg = True
                    for d in range(n + 1):
                        if deg_masks[d] == 0:
                            continue
                        Mu_d = mask_u_ex_v & deg_masks[d]
                        if Mu_d == 0:
                            continue
                        Mv_d = mask_v_ex_u & deg_masks[d]
                        if Mv_d == 0:
                            continue
                        
                        # Both non-empty. Check if they are identical singletons.
                        common = Mu_d & Mv_d
                        if common == Mu_d and common == Mv_d:
                            # They are the same set.
                            if common & (common - 1) == 0:
                                # Single bit set.
                                # s == t is the only pair.
                                continue
                        # Otherwise, violation
                        return False
                mask_u >>= 1
                v += 1
        return True

    def update_masks(adj_masks, deg_masks, edge, add):
        """
        Updates adj_masks and deg_masks in place for adding/removing an edge.
        Returns a copy of the previous state if needed for rollback? 
        Better to pass copies or store state.
        Here we will create new lists to keep it safe and simple, or modify and revert.
        Given the speed, modifying in place and reverting is best.
        """
        u, v = edge
        if add:
            # Increment degrees
            deg_u = adj_masks[u].bit_count()
            deg_v = adj_masks[v].bit_count()
            
            # Update deg_masks
            deg_masks[deg_u] &= ~(1 << u)
            deg_masks[deg_u + 1] |= (1 << u)
            
            deg_masks[deg_v] &= ~(1 << v)
            deg_masks[deg_v + 1] |= (1 << v)
            
            # Update adj_masks
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            # Decrement degrees
            deg_u = adj_masks[u].bit_count()
            deg_v = adj_masks[v].bit_count()
            
            # Update deg_masks
            deg_masks[deg_u] &= ~(1 << u)
            deg_masks[deg_u - 1] |= (1 << u)
            
            deg_masks[deg_v] &= ~(1 << v)
            deg_masks[deg_v - 1] |= (1 << v)
            
            # Update adj_masks
            adj_masks[u] &= ~(1 << v)
            adj_masks[v] &= ~(1 << u)

    def add_edge(adj_masks, deg_masks, edge):
        update_masks(adj_masks, deg_masks, edge, True)
        
    def remove_edge(adj_masks, deg_masks, edge):
        update_masks(adj_masks, deg_masks, edge, False)

    # Initial State: K_{7,8}
    # Partition A: 0..6, B: 7..14
    adj_masks = [0] * n
    deg_masks = [0] * (n + 1)
    
    for a in range(7):
        for b in range(7, 15):
            adj_masks[a] |= (1 << b)
            adj_masks[b] |= (1 << a)
            
    for i in range(n):
        d = adj_masks[i].bit_count()
        deg_masks[d] |= (1 << i)
        
    # Verify initial state
    if not check_valid(adj_masks, deg_masks):
        # Fallback to empty if logic error (should not happen)
        adj_masks = [0] * n
        deg_masks = [0] * (n + 1)

    best_adj = [m for m in adj_masks]
    best_deg = [m for m in deg_masks]
    best_count = sum(m.bit_count() for m in best_adj) // 2
    
    # Simulated Annealing / Local Search
    T = 50.0
    cooling = 0.995
    min_T = 0.1
    max_steps = 20000
    
    # Pre-calculate list of all edges for random selection
    # We need to pick random edges to add/remove.
    
    steps = 0
    while time.time() < deadline:
        # Pick a random move
        # 1. Add a random edge
        # 2. Remove a random edge
        # 3. Swap (Remove one, Add one)
        
        move_type = random.random()
        
        # Copy current state for rollback
        cur_adj = [m for m in adj_masks]
        cur_deg = [m for m in deg_masks]
        cur_count = sum(m.bit_count() for m in cur_adj) // 2
        
        if move_type < 0.3:
            # Add edge
            # Find a random edge not present
            # To avoid O(N) scan, we can just pick random u, v and check
            # But we need to ensure we try diverse edges.
            # Simple approach: pick random edge from all_edges
            e = random.choice(all_edges)
            if adj_masks[e[0]] & (1 << e[1]):
                continue # Already present
            
            add_edge(cur_adj, cur_deg, e)
            new_count = cur_count + 1
            new_valid = check_valid(cur_adj, cur_deg)
            new_score = new_count if new_valid else -1000
            
            delta = new_score - (cur_count if check_valid(adj_masks, deg_masks) else -1000)
            # Actually, we should track current score properly.
            # Let's assume current state is valid?
            # If current state is invalid, we should try to fix it.
            # Strategy: If current is valid, score = count. If invalid, score = count - 100 * violations?
            # We don't count violations easily with bitmask.
            # Let's stick to: Only accept moves that result in valid states, 
            # UNLESS we are stuck, in which case we allow invalid states with penalty?
            # 
            # Better: 
            # If current is valid:
            #   Accept if new is valid and new_count > cur_count.
            #   Accept if new is valid and new_count == cur_count with prob exp(-delta/T)? No delta is 0.
            #   Accept if new is invalid? No, we want to stay in valid space if possible.
            #   But to escape local optima, we might need to go invalid.
            #   
            # Let's use a score function:
            # score = count - 10 * num_violations (approx)
            # But we don't have num_violations.
            # 
            # Alternative:
            # If current is valid:
            #   Try to add. If valid, accept.
            #   If invalid, reject (or accept with very low prob to explore?).
            #   Try to swap. If valid, accept if count doesn't decrease?
            #   
            # Let's try a different approach:
            # Maintain a set of valid states.
            # Move: 
            #   1. Add edge. If valid, accept.
            #   2. Remove edge. If valid, accept (low prob, to explore).
            #   3. Swap. If valid, accept.
            
            # Let's restart the loop logic with a clearer strategy.
            # Current state is assumed valid.
            # If not, we are in trouble.
            
            # Let's check if current is valid.
            pass # Placeholder
            
        # Let's restructure the main loop to be cleaner.
        break # Break out of this messy draft to rewrite properly

    # REWRITING MAIN LOOP CLEANLY
    
    # State: adj_masks, deg_masks. Assumed valid.
    # Score: edge count.
    
    # Move 1: Add
    # Move 2: Swap (Remove 1, Add 1)
    # Move 3: 2-Swap (Remove 2, Add 2)
    
    # We will use a "temperature" to accept worse valid states? 
    # No, we maximize count. So we only accept if count increases (for add) 
    # or count stays same (for swap, to explore) or count increases.
    # If we only accept improvements, we get stuck.
    # So we accept equal-count swaps with probability p.
    
    # What if we are stuck at 56?
    # We need to find a way to get to 57.
    # This requires a sequence of moves that might temporarily decrease count?
    # Or a swap that keeps count same but changes structure to allow future adds?
    
    # Let's try:
    # 1. Random Add. If valid, accept.
    # 2. Random Swap (Remove e1, Add e2). If valid, accept if count >= current count (always true for swap).
    #    Actually swap keeps count same. So we accept with prob 0.5 to explore.
    # 3. Random 2-Swap. If valid, accept.
    
    # If we can't add, we swap to change structure.
    
    # Implementation:
    
    best_adj = [m for m in adj_masks]
    best_deg = [m for m in deg_masks]
    best_count = sum(m.bit_count() for m in best_adj) // 2
    
    # Precompute list of present edges for removal
    # We can just pick random u, v and check presence.
    
    steps = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    while time.time() < deadline:
        steps += 1
        if steps > max_steps:
            break
            
        cur_count = sum(m.bit_count() for m in adj_masks) // 2
        
        # Choose move
        r = random.random()
        
        # Save state
        old_adj = [m for m in adj_masks]
        old_deg = [m for m in deg_masks]
        
        if r < 0.5:
            # Add
            # Pick random edge
            e = random.choice(all_edges)
            if adj_masks[e[0]] & (1 << e[1]):
                continue # Already there
            
            add_edge(adj_masks, deg_masks, e)
            if check_valid(adj_masks, deg_masks):
                new_count = cur_count + 1
                # Accept
                if new_count > best_count:
                    best_count = new_count
                    best_adj = [m for m in adj_masks]
                    best_deg = [m for m in deg_masks]
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
            else:
                # Revert
                adj_masks = old_adj
                deg_masks = old_deg
                no_improve_steps += 1
                
        elif r < 0.8:
            # Swap: Remove one, Add one
            # Pick edge to remove
            # Find a random present edge
            # Pick random u, v
            removed = False
            for _ in range(10): # Try few times to find present edge
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if adj_masks[u] & (1 << v):
                    e_rem = (u, v)
                    removed = True
                    break
            
            if not removed:
                continue
            
            # Pick edge to add
            e_add = random.choice(all_edges)
            if e_add == e_rem:
                continue
            if adj_masks[e_add[0]] & (1 << e_add[1]):
                # Already present, this is not a swap, it's a no-op or invalid
                continue
            
            remove_edge(adj_masks, deg_masks, e_rem)
            add_edge(adj_masks, deg_masks, e_add)
            
            if check_valid(adj_masks, deg_masks):
                new_count = cur_count
                # Accept with probability 0.5 to explore
                if random.random() < 0.5:
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = [m for m in adj_masks]
                        best_deg = [m for m in deg_masks]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    # Revert
                    adj_masks = old_adj
                    deg_masks = old_deg
            else:
                # Revert
                adj_masks = old_adj
                deg_masks = old_deg
                no_improve_steps += 1
                
        else:
            # 2-Swap: Remove 2, Add 2
            # Find 2 present edges
            e_rem1 = None
            e_rem2 = None
            for _ in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if u > v: u, v = v, u
                if adj_masks[u] & (1 << v):
                    if e_rem1 is None:
                        e_rem1 = (u, v)
                    elif (u, v) != e_rem1:
                        e_rem2 = (u, v)
                        break
            
            if e_rem1 is None or e_rem2 is None:
                continue
            
            e_add1 = random.choice(all_edges)
            e_add2 = random.choice(all_edges)
            
            # Ensure adds are distinct and not the removed ones
            if e_add1 == e_add2 or e_add1 in [e_rem1, e_rem2] or e_add2 in [e_rem1, e_rem2]:
                continue
            
            # Check if adds are already present? 
            # If they are, we are effectively removing 1 and adding 1? 
            # No, we remove 2. If add1 is present, we are just removing e_rem1 and e_rem2, 
            # and "adding" e_add1 which is already there? That would be a double add?
            # We must ensure e_add1 and e_add2 are NOT present.
            
            if adj_masks[e_add1[0]] & (1 << e_add1[1]):
                continue
            if adj_masks[e_add2[0]] & (1 << e_add2[1]):
                continue
            
            remove_edge(adj_masks, deg_masks, e_rem1)
            remove_edge(adj_masks, deg_masks, e_rem2)
            add_edge(adj_masks, deg_masks, e_add1)
            add_edge(adj_masks, deg_masks, e_add2)
            
            if check_valid(adj_masks, deg_masks):
                new_count = cur_count
                if random.random() < 0.5:
                    if new_count > best_count:
                        best_count = new_count
                        best_adj = [m for m in adj_masks]
                        best_deg = [m for m in deg_masks]
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1
                else:
                    adj_masks = old_adj
                    deg_masks = old_deg
            else:
                adj_masks = old_adj
                deg_masks = old_deg
                no_improve_steps += 1
        
        # Cooling
        T *= cooling
        if T < min_T:
            T = min_T
            
        if no_improve_steps > max_no_improve:
            # Restart from best
            adj_masks = [m for m in best_adj]
            deg_masks = [m for m in best_deg]
            no_improve_steps = 0
            T = 10.0

    # Extract edges from best_adj
    edges = []
    for u in range(n):
        mask = best_adj[u]
        v = 0
        while mask:
            if mask & 1:
                if u < v:
                    edges.append([u, v])
            mask >>= 1
            v += 1
            
    return edges

# EVOLVE-BLOCK-END
