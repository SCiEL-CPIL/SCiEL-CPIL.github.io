import random
import time

def construct():
    n = 40
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # Initialize adjacency as list of bitmasks
    adj = [0] * n
    
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        
    def calc_n2():
        """Calculate neighbors at distance 2 for all nodes."""
        n2 = [0] * n
        for u in range(n):
            mask = 0
            # Iterate neighbors of u
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                mask |= adj[v]
                temp -= lsb
            # Remove u and direct neighbors
            mask &= ~adj[u]
            mask &= ~(1 << u)
            n2[u] = mask
        return n2
        
    def creates_c6(u, v, n2):
        """Check if adding edge (u,v) creates a C6."""
        # A C6 is created if there is a path of length 5 between u and v.
        # Path: u - a - b - c - d - v
        # b is in N2(u), c is in N2(v), and b~c.
        # So we check if (Union of N(b) for b in N2(u)) intersects N2(v).
        
        # Optimization: If u and v are adjacent, we don't add. 
        # This function is called for non-adjacent u, v.
        
        # Get N2(u) and N2(v)
        mask_u = n2[u]
        mask_v = n2[v]
        
        if mask_u == 0 or mask_v == 0:
            return False
            
        # Compute S = Union of N(b) for all b in N2(u)
        s_mask = 0
        temp = mask_u
        while temp:
            lsb = temp & -temp
            b = lsb.bit_length() - 1
            s_mask |= adj[b]
            temp -= lsb
            
        # Check intersection with N2(v)
        if s_mask & mask_v:
            return True
        return False

    def is_c6_free(n2):
        """Check if the current graph is C6-free."""
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] >> v) & 1:
                    if creates_c6(u, v, n2):
                        return False
        return True

    # 1. Initialization: Random Greedy
    # Start with empty graph
    # Try to add edges in random order
    edges_to_try = [(u, v) for u in range(n) for v in range(u + 1, n)]
    random.shuffle(edges_to_try)
    
    n2 = calc_n2()
    edge_count = 0
    
    for u, v in edges_to_try:
        if time.time() > deadline:
            break
        if creates_c6(u, v, n2):
            continue
        add_edge(u, v)
        edge_count += 1
        n2 = calc_n2()
        
    # 2. Local Search
    # We will try to add edges and swap edges to improve structure
    # To speed up, we keep a list of missing edges
    
    # Generate list of missing edges
    missing_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if not (adj[u] >> v) & 1:
                missing_edges.append((u, v))
                
    # To make it faster, we can just pick random pairs
    # But maintaining a list is better for "add" phase
    
    # We'll use a simple random walk in the space of graphs
    # State: adj, n2, edge_count
    
    while time.time() < deadline:
        # Try to add a random missing edge
        if missing_edges:
            idx = random.randint(0, len(missing_edges) - 1)
            u, v = missing_edges[idx]
            if not creates_c6(u, v, n2):
                # Add edge
                add_edge(u, v)
                edge_count += 1
                n2 = calc_n2()
                missing_edges.pop(idx)
                # Note: Adding an edge might make other "missing" edges invalid? 
                # No, it only makes it harder to add *more* edges.
                # But it doesn't turn existing edges into missing ones.
                # However, the list `missing_edges` is now stale if we don't update it?
                # We only remove the one we added. Other missing edges are still missing.
                # But some might now be invalid to add. We check validity before adding.
                continue
        
        # If no edges to add (or we want to explore), try a swap
        # Pick a random existing edge to remove
        existing_edges = []
        for u in range(n):
            temp = adj[u]
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                if u < v:
                    existing_edges.append((u, v))
                temp -= lsb
        
        if not existing_edges:
            break
            
        x, y = random.choice(existing_edges)
        
        # Pick a random missing edge to add
        # We need a valid missing edge. If missing_edges is empty, we are complete (impossible for C6-free?)
        # Or we just pick a random pair (a,b) that is not an edge.
        a, b = -1, -1
        tries = 0
        while a == -1 and tries < 10:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)
            if a == b or (adj[a] >> b) & 1:
                a, b = -1, -1
                tries += 1
            else:
                if a > b: a, b = b, a
        
        if a == -1:
            continue
            
        # Try swap: remove (x,y), add (a,b)
        # 1. Remove (x,y)
        remove_edge(x, y)
        n2_new = calc_n2()
        
        # 2. Check if adding (a,b) creates C6
        if not creates_c6(a, b, n2_new):
            # 3. Accept swap
            add_edge(a, b)
            # edge_count remains same
            n2 = n2_new
            # Update missing_edges list?
            # (x,y) is now missing. (a,b) is now present.
            # This is getting messy to maintain the list perfectly.
            # Let's just rebuild missing_edges occasionally or just rely on random pairs.
            # Given n=40, rebuilding is cheap.
            missing_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] >> v) & 1:
                        missing_edges.append((u, v))
        else:
            # 4. Reject swap, restore (x,y)
            add_edge(x, y)
            # n2 is already recalculated as n2_new, but we rejected, so we need original n2?
            # We can just recalc n2 to be safe, or keep the old one if we didn't modify adj permanently.
            # We did modify adj (remove then add back).
            # n2_new was calculated after remove.
            # We added back (x,y). So we need to recalc n2 again?
            # Or we can just keep the old n2 if we ensure we revert correctly.
            # Simplest: recalc n2.
            n2 = calc_n2()

    # Final verification (optional but good for debugging)
    # n2_final = calc_n2()
    # if not is_c6_free(n2_final):
    #     pass # Should not happen
    
    # Convert adj to list of edges
    final_edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if (adj[u] >> v) & 1:
                final_edges.append([u, v])
                
    return final_edges

if __name__ == "__main__":
    # Quick test
    edges = construct()
    print(len(edges))
