import random
import time

def construct():
    n = 20
    deadline = time.time() + 110.0  # Safety margin
    
    # Helper for bit count
    try:
        (1).bit_count()
        def popcount(x):
            return x.bit_count()
    except AttributeError:
        def popcount(x):
            return bin(x).count('1')

    def is_valid(u, v, adj):
        # Check if adding (u,v) violates K2,3
        # Adding (u,v) increases common neighbors for pairs (u,w) and (v,w) where w is neighbor of the other.
        # Specifically:
        # For any w != u, v:
        # If v ~ w, then N(u) and N(w) both gain v? No.
        # N(u) gains v. N(w) is unchanged.
        # Common neighbors of u and w: N(u) cap N(w).
        # Before: N(u) didn't have v. N(w) might have v.
        # If v in N(w), then v was already in N(w). Was v in N(u)? No, we are adding u-v.
        # So v was NOT in N(u).
        # So v was NOT in N(u) cap N(w).
        # After adding u-v, v IS in N(u).
        # So if v in N(w), then v becomes a common neighbor of u and w.
        # So |N(u) cap N(w)| increases by 1 if v in N(w).
        # We require |N(u) cap N(w)| <= 2 after addition.
        # So we need |N(u) cap N(w)| before addition <= 1 if v in N(w).
        
        for w in range(n):
            if w == u or w == v:
                continue
            
            # Check pair (u, w)
            if (adj[v] >> w) & 1:
                # v is neighbor of w
                # v will become common neighbor of u and w
                common = adj[u] & adj[w]
                if popcount(common) > 1:
                    return False
            
            # Check pair (v, w)
            if (adj[u] >> w) & 1:
                # u is neighbor of w
                # u will become common neighbor of v and w
                common = adj[v] & adj[w]
                if popcount(common) > 1:
                    return False
        
        return True

    def add_edge(u, v, adj):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(u, v, adj):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def count_edges(adj):
        deg_sum = 0
        for i in range(n):
            deg_sum += popcount(adj[i])
        return deg_sum // 2

    def check_all(adj):
        for i in range(n):
            for j in range(i+1, n):
                common = adj[i] & adj[j]
                if popcount(common) > 2:
                    return False
        return True

    best_edges = 0
    best_adj = [0] * n
    
    # Fallback: A simple valid graph (e.g., a cycle or empty)
    # Let's use a random valid graph as initial best if search fails
    # But we will run search.
    
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    
    while time.time() < deadline:
        adj = [0] * n
        edges = []
        
        # Random shuffle
        random.shuffle(all_edges)
        
        # Greedy Add
        for u, v in all_edges:
            if is_valid(u, v, adj):
                add_edge(u, v, adj)
                edges.append((u, v))
        
        m = len(edges)
        
        # Local Search: Try to add any missing edge
        # We can re-iterate over non-edges
        # To speed up, we can just check all pairs not in adj
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(n):
                for j in range(i+1, n):
                    if not (adj[i] >> j) & 1:
                        if is_valid(i, j, adj):
                            add_edge(i, j, adj)
                            edges.append((i, j))
                            m += 1
                            improved = True
                            if time.time() > deadline:
                                break
                if time.time() > deadline:
                    break
        
        # Local Search: 2-Opt (Remove 1, Add 2)
        # This is more complex. Let's try a simple version:
        # Pick a random edge, remove it, try to add 2 random non-edges.
        # If successful, keep. Else, restore.
        
        if m > best_edges:
            best_edges = m
            best_adj = adj[:]
            
        # If we found a very good solution, we can break early?
        # Theoretical max is unknown, but let's keep searching.
        
    # Convert best_adj to list of edges
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                final_edges.append([i, j])
                
    # Ensure we return at least a valid graph
    if not final_edges:
        # Return a simple cycle
        for i in range(n):
            final_edges.append([i, (i+1)%n])
            
    return final_edges
