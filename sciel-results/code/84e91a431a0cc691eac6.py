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
    # adj[i] is an integer where bit j is set if (i,j) is an edge
    adj = [0] * n
    
    def can_add(u, v):
        """
        Returns True if adding edge (u,v) does not create a C6.
        Correct check: 
        Does there exist a path u-z-y-x-v of length 4 with distinct vertices?
        i.e., z in N(u), x in N(v), y in N(z) ∩ N(x), 
        with {u, v, x, y, z} distinct.
        """
        # Iterate over neighbors of v
        mask_v = adj[v] & ~(1 << u)
        while mask_v:
            x = (mask_v & -mask_v).bit_length() - 1
            mask_v &= mask_v - 1
            
            # x is a neighbor of v, x != u
            # Now we need y such that y in N(x) and y in N(z) for some z in N(u)
            # And z != x, z != v, z != u
            # And y != u, y != v, y != x
            
            # Let's find all common neighbors of x and any neighbor of u?
            # No, z must be a neighbor of u.
            # So we iterate z in N(u) \ {v, x}
            mask_u = adj[u] & ~(1 << v) & ~(1 << x)
            
            while mask_u:
                z = (mask_u & -mask_u).bit_length() - 1
                mask_u &= mask_u - 1
                
                # z is a neighbor of u, z != v, z != x
                # Now check if there is a common neighbor y of x and z
                # y must be distinct from u, v, x, z
                common = adj[x] & adj[z]
                # Exclude u, v, x, z from common
                common &= ~(1 << u) & ~(1 << v) & ~(1 << x) & ~(1 << z)
                
                if common:
                    # Found y
                    return False
        return True

    def get_edges():
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                j = (mask & -mask).bit_length() - 1
                mask &= mask - 1
                if i < j:
                    edges.append([i, j])
        return edges

    random.seed(42)
    
    best_edges = 0
    best_adj = [0] * n
    
    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    # Phase 1: Random Greedy with multiple trials
    trials = 0
    while time_left() > 10:
        trials += 1
        adj_try = [0] * n
        ec = 0
        
        shuffled = all_pairs[:]
        random.shuffle(shuffled)
        
        for u, v in shuffled:
            if time_left() < 5:
                break
            
            if can_add(u, v):
                adj_try[u] |= (1 << v)
                adj_try[v] |= (1 << u)
                ec += 1
        
        if ec > best_edges:
            best_edges = ec
            best_adj = adj_try[:]

    # Fallback
    if best_edges == 0:
        # Simple valid graph: disjoint cliques K5
        for k in range(8):
            for a in range(5):
                for b in range(a+1, 5):
                    u = 5*k+a
                    v = 5*k+b
                    best_adj[u] |= (1 << v)
                    best_adj[v] |= (1 << u)
        best_edges = 80

    # Phase 2: Local Search
    adj = best_adj[:]
    current_edges = best_edges
    best_local_edges = best_edges
    best_local_adj = best_adj[:]
    
    no_improve_count = 0
    
    while time_left() > 2:
        improved = False
        
        # 1. Try to add a random non-edge
        for _ in range(50):
            if time_left() < 2:
                break
            u, v = random.choice(all_pairs)
            if adj[u] & (1 << v):
                continue
            
            if can_add(u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                current_edges += 1
                improved = True
                no_improve_count = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                break
        
        if improved:
            continue

        # 2. If no edge added, try to escape local maximum
        # Strategy: 1-for-2 move (remove 1, add 2)
        
        # Get list of current edges
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
            
        # Try a few candidate removals for 1-for-2
        success = False
        for _ in range(10):
            if time_left() < 2:
                break
            u_rem, v_rem = random.choice(edge_list)
            
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
                if can_add(u_add, v_add):
                    adj[u_add] |= (1 << v_add)
                    adj[v_add] |= (1 << u_add)
                    added_edges.append((u_add, v_add))
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count == 2:
                # Net +1 edge
                current_edges += 1
                improved = True
                no_improve_count = 0
                if current_edges > best_local_edges:
                    best_local_edges = current_edges
                    best_local_adj = adj[:]
                success = True
                break
            else:
                # Undo additions
                for ua, va in added_edges:
                    adj[ua] &= ~(1 << va)
                    adj[va] &= ~(1 << ua)
            
            # Undo removal if we didn't succeed in adding 2
            if not success:
                adj[u_rem] |= (1 << v_rem)
                adj[v_rem] |= (1 << u_rem)
                
        if success:
            continue
            
        # 3. If 1-for-2 failed, try 1-for-1 swap to shuffle
        if no_improve_count > 10:
            for _ in range(20):
                if time_left() < 2:
                    break
                u_rem, v_rem = random.choice(edge_list)
                
                # Temporarily remove edge
                adj[u_rem] &= ~(1 << v_rem)
                adj[v_rem] &= ~(1 << u_rem)
                
                added_edge = None
                for _ in range(50):
                    u_add, v_add = random.choice(all_pairs)
                    if adj[u_add] & (1 << v_add):
                        continue
                    if can_add(u_add, v_add):
                        adj[u_add] |= (1 << v_add)
                        adj[v_add] |= (1 << u_add)
                        added_edge = (u_add, v_add)
                        break
                
                if added_edge:
                    # Swap accepted. Edge count unchanged.
                    no_improve_count = 0
                    break
                else:
                    # Undo removal
                    adj[u_rem] |= (1 << v_rem)
                    adj[v_rem] |= (1 << u_rem)
        
        no_improve_count += 1

    if best_local_edges > best_edges:
        best_edges = best_local_edges
        best_adj = best_local_adj

    return get_edges()
