import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    best_edges = []
    best_count = 0
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, cc, u, v):
        # Check if edge (u,v) already exists
        if (adj[u] >> v) & 1:
            return False
        
        # Check constraints for neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # So we need current common(v, w) <= 1.
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            if cc[v][w] > 1:
                return False
            w_mask ^= lsb
            
        # Check constraints for neighbors of v
        # For w in N(v), v becomes a common neighbor of u and w.
        # So we need current common(u, w) <= 1.
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            if cc[u][w] > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, cc, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
        # Update common counts for neighbors of u
        w_mask = adj[u] & ~(1 << v) # Exclude v itself if it was just added? 
        # Wait, adj[u] already has v set. 
        # We need to iterate over OLD neighbors of u.
        # Let's pass old_adj or calculate before setting.
        # In this implementation, we call add_edge AFTER is_valid_add.
        # is_valid_add used the old adj.
        # So we should iterate over the neighbors BEFORE adding u-v.
        # But adj[u] is already updated in the first two lines.
        # Let's fix add_edge to take care of this.
        
        # Actually, it's safer to update adj AFTER iterating, or pass a mask.
        # Let's refactor: is_valid_add uses adj. add_edge modifies adj.
        # If I modify adj[u] first, I lose the old neighbor list.
        
        # Let's swap order:
        # 1. Iterate old neighbors of u
        # 2. Iterate old neighbors of v
        # 3. Update adj
        
        # But I already wrote add_edge to update adj first.
        # Let's rewrite add_edge properly.
        
        pass # Placeholder, see below

    def add_edge_correct(adj, cc, u, v):
        # Update common counts based on OLD neighbors
        # Old neighbors of u: adj[u] (before adding v)
        # Old neighbors of v: adj[v] (before adding u)
        
        # Since we call this after is_valid_add, adj is still in old state?
        # No, is_valid_add doesn't modify adj.
        # So adj is old state.
        
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            cc[v][w] += 1
            cc[w][v] += 1
            w_mask ^= lsb
            
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            cc[u][w] += 1
            cc[w][u] += 1
            w_mask ^= lsb
            
        # Now add the edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge_correct(adj, cc, u, v):
        # Remove edge first
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
        # Now adj[u] and adj[v] are in the state AFTER removal (which is the state we want to revert to? No.)
        # We want to decrement counts for the neighbors that WERE connected to u/v.
        # The neighbors that were connected are the current adj[u] and adj[v] (after removal).
        # Wait.
        # Before removal: adj_old[u] = adj_new[u] | {v}.
        # The common count increase was caused by u being added to N(v).
        # So for each w in N_old(v), cc[u][w] increased.
        # N_old(v) = adj_new[v] | {u}.
        # But u is not a neighbor of u.
        # So we need to iterate over N_old(v) which is adj_new[v] plus u?
        # No, the update was:
        # For w in N_old(u): cc[v][w] += 1.
        # For w in N_old(v): cc[u][w] += 1.
        
        # So to revert:
        # For w in N_old(u): cc[v][w] -= 1.
        # N_old(u) is the current adj[u] (after removing v) PLUS v?
        # No, v is a neighbor of u.
        # The loop in add_edge iterated over adj[u] (which did NOT contain v).
        # So we should iterate over the current adj[u] (which does NOT contain v) to decrement?
        # Yes. The set of w's is exactly the current adj[u].
        
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            cc[v][w] -= 1
            cc[w][v] -= 1
            w_mask ^= lsb
            
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            cc[u][w] -= 1
            cc[w][u] -= 1
            w_mask ^= lsb

    def get_missing(adj):
        missing = []
        for i in range(n):
            mask = ~adj[i] & ((1 << n) - 1)
            mask &= ~((1 << (i + 1)) - 1) # j > i
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                missing.append((i, j))
                mask ^= lsb
        return missing

    def get_present(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            mask &= ~((1 << (i + 1)) - 1) # j > i
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, cc, count):
        while time.time() < deadline:
            missing = get_missing(adj)
            if not missing:
                break
            random.shuffle(missing)
            improved = False
            for u, v in missing:
                if is_valid_add(adj, cc, u, v):
                    add_edge_correct(adj, cc, u, v)
                    count += 1
                    improved = True
            if not improved:
                break
        return count

    def local_search_with_swaps(adj, cc, count, max_iter=50):
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            missing = get_missing(adj)
            present = get_present(adj)
            
            if not present:
                break
                
            # Try to find a swap: remove 1, add 1 or 2
            # We just pick a random present edge to remove, then try to fill
            # This is effectively a kick.
            
            # Better: Try to add a missing edge. If fail, try to remove a present edge that "blocks" it.
            # For simplicity, let's just do random kick + greedy fill.
            
            # Random Kick: Remove 1-3 random edges
            kick_size = random.randint(1, 3)
            edges_to_remove = random.sample(present, kick_size)
            
            for u, v in edges_to_remove:
                remove_edge_correct(adj, cc, u, v)
                count -= 1
                
            # Greedy fill
            count = greedy_fill(adj, cc, count)
            
        return count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 10:
        # Initialize
        adj = [0] * n
        cc = [[0] * n for _ in range(n)]
        count = 0
        
        # Random Greedy Start
        random_edges = all_edges[:]
        random.shuffle(random_edges)
        for u, v in random_edges:
            if is_valid_add(adj, cc, u, v):
                add_edge_correct(adj, cc, u, v)
                count += 1
        
        # Local Search
        count = greedy_fill(adj, cc, count)
        
        # Simulated Annealing / Random Restarts within the run
        T = 1.0
        while T > 0.05 and time.time() < deadline - 5:
            # Try to improve
            # 1. Try to add a random missing edge
            if random.random() < 0.7:
                missing = get_missing(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, cc, u, v):
                        add_edge_correct(adj, cc, u, v)
                        count += 1
            else:
                # Kick: Remove random edge
                present = get_present(adj)
                if present:
                    u, v = random.choice(present)
                    remove_edge_correct(adj, cc, u, v)
                    count -= 1
                    # Try to fill back
                    count = greedy_fill(adj, cc, count)
            
            T *= 0.99
            
        if count > best_count:
            best_count = count
            best_edges = []
            for i in range(n):
                mask = adj[i]
                while mask:
                    lsb = mask & -mask
                    j = lsb.bit_length() - 1
                    if i < j:
                        best_edges.append([i, j])
                    mask ^= lsb
                    
        restarts += 1
        
    return best_edges
