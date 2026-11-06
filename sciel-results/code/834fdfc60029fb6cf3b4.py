import random
import time
import math

def construct():
    n = 31
    m = 91
    deadline = time.time() + 115  # Safety margin
    
    # Bitset representation of adjacency matrix
    # adj[i] is an integer where bit j is 1 if edge (i,j) exists
    adj = [0] * n
    
    # Helper to get common neighbors count
    def common_neighbors(u, v):
        # adj[u] & adj[v] gives a bitmask of common neighbors
        # We need to count bits
        return (adj[u] & adj[v]).bit_count()
    
    # Helper to get number of C4s added by edge (u,v)
    def c4_cost(u, v):
        cn = common_neighbors(u, v)
        return cn * (cn - 1) // 2
        
    # Initialize random graph
    # We need exactly m edges
    # Generate all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
    random.shuffle(all_edges)
    
    # Take first m edges
    for u, v in all_edges[:m]:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    # Current C4 count
    current_c4 = sum(c4_cost(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1)
    
    best_c4 = current_c4
    best_adj = adj[:]
    
    # Local search
    # We need a list of current edges and non-edges for fast selection
    # Or just pick random pairs and check
    
    # To make it faster, maintain a list of edges
    current_edges = set(all_edges[:m])
    
    # Simulated Annealing / Random Walk
    T = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Pick random edge to remove
        if not current_edges: break
        u, v = random.choice(list(current_edges))
        
        # Pick random non-edge to add
        # We need to find a non-edge. 
        # Picking random (w,x) and checking if it's in current_edges is O(1) amortized if we try a few times.
        # But n=31, total edges ~465. 91/465 is ~20%. So 80% chance of picking a non-edge.
        
        w = random.randint(0, n-1)
        x = random.randint(0, n-1)
        if w == x: continue
        if w > x: w, x = x, w
        
        if (w, x) in current_edges:
            continue
            
        # Calculate delta
        # Remove (u,v): decrease by c4_cost(u,v) in current graph
        # Add (w,x): increase by c4_cost(w,x) in current graph
        # Note: c4_cost depends on adj. 
        # The removal of (u,v) changes adj, which might affect c4_cost(w,x) if {u,v} intersects {w,x} neighbors?
        # No, c4_cost(w,x) depends on common neighbors of w and x.
        # Removing (u,v) only changes adj[u] and adj[v].
        # If w or x is u or v, the common neighbor count changes.
        
        # Let's calculate carefully.
        # State 1: Current graph G.
        # State 2: G - (u,v) + (w,x).
        
        # Cost in G for (u,v) is C_uv = binom(N_G(u,v), 2).
        # Cost in G for (w,x) is C_wx = binom(N_G(w,x), 2).
        
        # In G - (u,v), let's call it G1.
        # The cost of adding (w,x) to G1 is binom(N_G1(w,x), 2).
        
        # If {u,v} is disjoint from {w,x} and no edges between them affect common neighbors... 
        # Actually, N_G1(w,x) = N_G(w,x) \ {u, v} ?
        # u is a common neighbor of w,x if u-w and u-x exist.
        # Removing (u,v) does not remove (u,w) or (u,x). So u remains a neighbor of w and x if it was.
        # So N_G1(w,x) is the same as N_G(w,x) UNLESS u or v is w or x?
        # If w=u, then we are removing (u,v). w=u.
        # Then we are adding (u,x).
        # This is a swap involving one vertex.
        
        # Simplification: 
        # Just compute the new C4 count? No, too slow.
        # Use the delta formula properly.
        
        # Let's just compute the change.
        # Old C4s involving (u,v): C_uv.
        # New C4s involving (w,x) in the new graph: 
        # We need N_new(w,x).
        
        # Let's just do a full recount if n is small? 
        # n=31. 465 pairs. 
        # Recount is 465 * bit_count. Very fast.
        # Let's just try the swap, recount, and accept if better.
        
        # Apply swap
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        current_edges.remove((u, v))
        
        adj[w] |= (1 << x)
        adj[x] |= (1 << w)
        current_edges.add((w, x))
        
        new_c4 = sum(c4_cost(i, j) for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1)
        
        if new_c4 < current_c4:
            current_c4 = new_c4
            if new_c4 < best_c4:
                best_c4 = new_c4
                best_adj = adj[:]
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            current_edges.add((u, v))
            
            adj[w] ^= (1 << x)
            adj[x] ^= (1 << w)
            current_edges.remove((w, x))
            
            # Annealing acceptance
            if new_c4 == current_c4:
                if random.random() < 0.1:
                    # Accept
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    current_edges.remove((u, v))
                    
                    adj[w] |= (1 << x)
                    adj[x] |= (1 << w)
                    current_edges.add((w, x))
                    current_c4 = new_c4
            else:
                # new_c4 > current_c4
                delta = new_c4 - current_c4
                if random.random() < math.exp(-delta / T):
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    current_edges.remove((u, v))
                    
                    adj[w] |= (1 << x)
                    adj[x] |= (1 << w)
                    current_edges.add((w, x))
                    current_c4 = new_c4
        
        T *= cooling
        if T < 0.01:
            T = 0.01

    # Convert best_adj back to list of edges
    final_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                final_edges.append([i, j])
                
    return final_edges
