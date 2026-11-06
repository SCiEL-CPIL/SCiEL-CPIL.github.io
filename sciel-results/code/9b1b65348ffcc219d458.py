import time
import random
import sys

def construct():
    n = 40
    start_time = time.time()
    deadline = 115.0  # 5s buffer for 120s limit
    
    def time_left():
        return deadline - (time.time() - start_time)
    
    # Bitset adjacency
    adj = [0] * n
    
    def can_add(u, v, adj_state):
        """
        Check if adding edge (u,v) creates a C6.
        C6: u-v-x-y-z-w-u
        u, v are the new edge.
        x in N(v), w in N(u).
        Need path x-y-z-w of length 2 (i.e., x~y, y~z, z~w) 
        such that y,z are distinct and not in {u,v,x,w}.
        """
        Nv = adj_state[v] & ~(1 << u)
        Nu = adj_state[u] & ~(1 << v)
        
        # Iterate over the smaller set to reduce outer loops
        if Nv.bit_count() > Nu.bit_count():
            Nv, Nu = Nu, Nv
            # Swap u and v roles? No, the structure is symmetric.
            # If we swap, x in N(u), w in N(v).
            # The check is symmetric. We just need to be careful with the masks.
            # Let's stick to: x from smaller set, w from larger set.
            # If Nv was smaller, x in Nv, w in Nu.
            # If Nu was smaller, we swapped, so now Nv is the smaller one (original Nu).
            # So x is from the smaller set, w from the larger.
            # The logic below assumes x is from the first arg, w from second.
            # Since we swapped, x is now from original Nu, w from original Nv.
            # This is fine as long as we mask correctly.
            # But wait, the masks Nv and Nu are local to this function.
            # We need to ensure we are checking the correct neighbors.
            # Let's not swap to avoid confusion, just pick the smaller for the outer loop.
            pass 
            
        # Let's implement without swap for clarity, relying on Python speed or small degrees
        # Max degree is ~10, so 10*10*10 = 1000 ops is fine.
        
        # Iterate x over N(v)
        x_mask = Nv
        while x_mask:
            x = (x_mask & -x_mask).bit_length() - 1
            x_mask &= x_mask - 1
            if x == u:
                continue
            # Iterate w over N(u)
            w_mask = Nu
            while w_mask:
                w = (w_mask & -w_mask).bit_length() - 1
                w_mask &= w_mask - 1
                if w == v or w == x:
                    continue
                # Need y in N(x) \ {u,v,w}, z in N(w) \ {u,v,x}
                # such that y~z, y!=z.
                # S_x = neighbors of x excluding u,v,w
                S_x = adj_state[x] & ~(1 << u) & ~(1 << v) & ~(1 << w)
                S_w = adj_state[w] & ~(1 << u) & ~(1 << v) & ~(1 << x)
                
                # Check if there is an edge between S_x and S_w
                # Iterate y in S_x
                y_mask = S_x
                while y_mask:
                    y = (y_mask & -y_mask).bit_length() - 1
                    y_mask &= y_mask - 1
                    # Check if y is connected to any z in S_w, z != y
                    # z_mask = S_w & ~(1 << y)
                    if adj_state[y] & (S_w & ~(1 << y)):
                        return False
        return True

    def get_edges(adj_state):
        edges = []
        for i in range(n):
            mask = adj_state[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    best_edges = 0
    best_adj = [0] * n
    
    # Phase 1: Multiple Random Greedy Trials
    # Use a heuristic: prioritize edges where sum of degrees is low? 
    # Or just random shuffle. Random shuffle is robust.
    
    for trial in range(50):
        if time_left() < 10:
            break
        
        adj_try = [0] * n
        ec = 0
        
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 2:
                break
            if can_add(u, v, adj_try):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    # Fallback
    if best_edges == 0:
        # Simple fallback: disjoint cliques
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 80

    # Phase 2: Local Search with 1-for-2 Moves
    adj = best_adj[:]
    current_edges = best_edges
    best_local_edges = best_edges
    best_local_adj = best_adj[:]
    
    # Tabu list for recent removals to prevent immediate re-addition
    tabu = {}
    tabu_iteration = 0
    
    while time_left() > 5:
        tabu_iteration += 1
        improved = False
        
        # 1. Try to add a random non-edge (0-for-1 move)
        # Limit attempts to avoid wasting time
        for _ in range(20):
            if time_left() < 5:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            if can_add(u, v, adj):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                improved = True
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                break
        
        if improved:
            continue

        # 2. Try 1-for-2 moves: Remove one edge, add two
        # Find a random edge to remove
        edge_list = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edge_list.append((i, j))
        
        if not edge_list:
            break
            
        # Try a few candidate removals
        for _ in range(5):
            if time_left() < 5:
                break
            u_rem, v_rem = random.choice(edge_list)
            
            # Check if this removal is tabu
            if tabu.get((u_rem, v_rem), 0) > tabu_iteration - 50:
                continue
            
            # Temporarily remove edge
            adj[u_rem] &= ~(1 << v_rem)
            adj[v_rem] &= ~(1 << u_rem)
            
            added_count = 0
            added_edges = []
            
            # Try to add two edges
            for _ in range(100):
                u_add, v_add = random.choice(all_pairs)
                if adj[u_add] & (1 << v_add):
                    continue
                if (u_add, v_add) in added_edges:
                    continue
                if can_add(u_add, v_add, adj):
                    adj[u_add] |= (1 << v_add)
                    adj[v_add] |= (1 << u_add)
                    added_edges.append((u_add, v_add))
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count >= 1:
                # Net gain: added_count - 1
                if added_count == 1:
                    # We removed 1, added 1. No gain.
                    # But maybe the new configuration allows further additions?
                    # Let's accept it as a move if we are not improving edge count immediately,
                    # but to keep it simple, we only accept if we gain net edges (added_count > 1)
                    # OR if we are stuck for a long time (annealing).
                    # For now, require added_count == 2 for a net +1.
                    pass
                else:
                    # Net +1 edge
                    current_edges += (added_count - 1)
                    tabu[(u_rem, v_rem)] = tabu_iteration
                    improved = True
                    if current_edges > best_local_edges:
                        best_local_edges = current_edges
                        best_local_adj = adj[:]
                    break
            else:
                # Undo removal
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
                # Undo additions (none added)
        
        if not improved:
            # If no improvement, we can stay put or try a random restart later
            # To avoid infinite loop, we do nothing specific, just loop again.
            # The tabu list helps avoid cycling on the same removals.
            pass

    if best_local_edges > best_edges:
        best_edges = best_local_edges
        best_adj = best_local_adj

    return get_edges(best_adj)
