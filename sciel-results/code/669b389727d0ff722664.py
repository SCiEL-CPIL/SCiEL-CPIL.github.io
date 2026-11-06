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
    
    def is_valid_add(adj, u, v):
        # adj is a list of integers (bitmasks)
        # 1. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 2. Check constraints on neighbors of u
        # For w in N(u), u becomes a common neighbor of v and w.
        # So we need current common(v, w) <= 1.
        w_mask = adj[u]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            # Calculate common neighbors of v and w
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 3. Check constraints on neighbors of v
        # For w in N(v), v becomes a common neighbor of u and w.
        # So we need current common(u, w) <= 1.
        w_mask = adj[v]
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if not (mask >> j) & 1:
                    missing.append((i, j))
        return missing

    def greedy_fill(order):
        adj = [0] * n
        count = 0
        for u, v in order:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def local_search(adj, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add any missing edge
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    # Don't break, keep adding if possible
                    # But we need to re-evaluate? No, adding an edge only makes constraints tighter,
                    # so other previously valid edges might become invalid?
                    # Wait. If we add (u,v), does it invalidate adding (u,w)?
                    # Yes, if w is a common neighbor of v and some other node, etc.
                    # Actually, adding an edge can only *decrease* the set of valid edges we can add later.
                    # So if we just added (u,v), we should restart the scan or be careful.
                    # However, since we are just checking `is_valid_add` on the current `adj`,
                    # and we update `adj` immediately, the check for the next edge in the loop
                    # will see the updated `adj`. So it is safe.
                    if time.time() > deadline:
                        return count
            
            # Phase 2: If no edges could be added, try a swap (remove 1, add 2)
            if not improved:
                present = []
                for i in range(n):
                    mask = adj[i]
                    while mask:
                        lsb = mask & -mask
                        j = lsb.bit_length() - 1
                        if i < j:
                            present.append((i, j))
                        mask ^= lsb
                
                if not present:
                    break
                
                random.shuffle(present)
                # Try a few random removals
                for _ in range(50):
                    if time.time() > deadline:
                        return count
                    
                    # Pick a random edge to remove
                    rem_edge = random.choice(present)
                    u, v = rem_edge
                    remove_edge(adj, u, v)
                    temp_count = count - 1
                    
                    # Try to add as many edges as possible
                    added_any = False
                    for _ in range(10):
                        if time.time() > deadline:
                            # Revert and return current best
                            add_edge(adj, u, v)
                            return count
                        missing = get_missing_edges(adj)
                        random.shuffle(missing)
                        for a, b in missing:
                            if is_valid_add(adj, a, b):
                                add_edge(adj, a, b)
                                temp_count += 1
                                added_any = True
                                break
                        
                        if not added_any:
                            break
                    
                    if temp_count > count:
                        count = temp_count
                        improved = True
                        # Update present list? No, we just break out of the swap loop
                        # and go back to Phase 1 in the main while loop
                        break
                    else:
                        # Revert
                        add_edge(adj, u, v)
                        # count is still original count
                        # We don't break, we try another removal?
                        # It's better to try a few, but if none work, we are stuck.
                        # We can break the swap loop if we've tried enough.
        
        return count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 5:
        random.shuffle(all_edges)
        adj, count = greedy_fill(all_edges)
        
        # Local search
        count = local_search(adj, count)
        
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
        
        # If we are running out of time, break early
        if time.time() > deadline - 10:
            break

    return best_edges
