import time
import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    deadline = time.time() + 115.0
    
    # Use bitmasks for fast intersection
    # adj[i] is an integer where bit j is 1 if (i,j) is an edge
    adj = [0] * n
    
    # Helper to get edge count
    def get_edge_count():
        return sum(bin(adj[i]).count('1') for i in range(n)) // 2

    # Helper to check edge
    def has_edge(u, v):
        return (adj[u] >> v) & 1

    # Helper to add/remove edge
    def add_edge(u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    def remove_edge(u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    # Compute book size for a specific edge (u, v)
    def book_size(u, v):
        if not has_edge(u, v):
            return 0
        common = adj[u] & adj[v]
        # Count bits
        return bin(common).count('1')

    # Compute max book size and the set of edges achieving it
    def compute_max_book():
        max_b = 0
        critical_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if has_edge(i, j):
                    b = book_size(i, j)
                    if b > max_b:
                        max_b = b
                        critical_edges = [(i, j)]
                    elif b == max_b:
                        critical_edges.append((i, j))
        return max_b, critical_edges

    # Initial construction: Random graph with 401 edges?
    # Or start with K_20,20 + 1 edge?
    # K_20,20 has 400 edges. Max book is 0 (bipartite).
    # Adding 1 edge creates triangles.
    # Let's start with a random graph to avoid bad local minima of bipartite structure?
    # No, bipartite is a good start because max book is low.
    # But we need 401 edges.
    
    # Start with K_20,20
    for i in range(20):
        for j in range(20, 40):
            add_edge(i, j)
            
    # Add 1 random edge within part A or B
    # To minimize book, pick an edge that creates fewest triangles?
    # In K_20,20, any intra-edge (u,v) in A has 20 common neighbors in B.
    # So book size is 20.
    # This is bad. We want to break these.
    
    # Better start: Random graph with average degree ~ 20.
    random.seed(42)
    adj = [0] * n
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                add_edge(i, j)
                edges.append((i, j))
    
    # Adjust edge count to exactly 401
    current_count = get_edge_count()
    while current_count < target_edges:
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and not has_edge(u, v):
            add_edge(u, v)
            current_count += 1
    while current_count > target_edges:
        # Pick a random edge to remove
        u = random.randint(0, n-1)
        v = random.randint(0, n-1)
        if u != v and has_edge(u, v):
            remove_edge(u, v)
            current_count -= 1

    # Local Search
    best_max = 999
    best_adj = adj.copy()
    
    # We will maintain the book sizes for all edges to speed up delta calculation?
    # Or just recompute affected ones.
    # Let's try a simple hill climber that recomputes max book from scratch every few steps,
    # but uses a fast "delta" check for the swap.
    
    # To do fast delta:
    # If we remove (u,v) and add (x,y):
    # 1. Removing (u,v) decreases book of (u,w) and (v,w) for all w in N(u) & N(v).
    # 2. Adding (x,y) increases book of (x,z) and (y,z) for all z in N(x) & N(y).
    # 3. It also affects the book of (u,v) itself (becomes 0) and (x,y) (becomes |N(x)&N(y)|).
    
    # This is complex to track globally.
    # Given n=40, E~400, full recompute is 400 * popcount(40-bit int).
    # popcount is very fast in Python 3.11+ (int.bit_count()).
    # 400 * ~10ns = 4000ns = 4us.
    # 120s / 4us = 30,000,000 steps.
    # We can do full recompute every step!
    
    # Let's verify bit_count speed.
    # Yes, int.bit_count() is C-level.
    
    current_max, _ = compute_max_book()
    
    no_improve = 0
    max_no_improve = 5000 # Allow some wandering
    
    # Precompute all possible edges and non-edges?
    # Or just pick random.
    
    while time.time() < deadline:
        if current_max <= 6:
            break
            
        # Pick a random edge to remove
        u, v = None, None
        while u is None:
            u = random.randint(0, n-1)
            v = random.randint(0, n-1)
            if u < v and has_edge(u, v):
                break
            else:
                u, v = None, None
                
        # Pick a random non-edge to add
        x, y = None, None
        while x is None:
            x = random.randint(0, n-1)
            y = random.randint(0, n-1)
            if x < y and not has_edge(x, y):
                break
            else:
                x, y = None, None
                
        if (u, v) == (x, y):
            continue
            
        # Perform swap
        remove_edge(u, v)
        add_edge(x, y)
        
        new_max, _ = compute_max_book()
        
        if new_max < current_max:
            current_max = new_max
            if current_max < best_max:
                best_max = current_max
                best_adj = adj.copy()
            no_improve = 0
        else:
            # Revert
            remove_edge(x, y)
            add_edge(u, v)
            no_improve += 1
            
            if no_improve > max_no_improve:
                # Try a more aggressive move: remove an edge from a critical triangle
                # or restart with new random seed?
                # Let's try to specifically break the max book edges.
                _, critical_edges = compute_max_book()
                if critical_edges:
                    cu, cv = random.choice(critical_edges)
                    common = adj[cu] & adj[cv]
                    if common:
                        # Pick a common neighbor w
                        # Find index of a set bit
                        w = 0
                        mask = common
                        while not (mask & 1):
                            mask >>= 1
                            w += 1
                        # Try removing (cu, w) or (cv, w)
                        # And add a random edge
                        if random.random() < 0.5:
                            rem_u, rem_v = cu, w
                        else:
                            rem_u, rem_v = cv, w
                            
                        # Find a non-edge to add
                        ax, ay = None, None
                        while ax is None:
                            ax = random.randint(0, n-1)
                            ay = random.randint(0, n-1)
                            if ax < ay and not has_edge(ax, ay):
                                break
                            else:
                                ax, ay = None, None
                                
                        if (rem_u, rem_v) == (ax, ay):
                            continue
                            
                        remove_edge(rem_u, rem_v)
                        add_edge(ax, ay)
                        
                        new_max, _ = compute_max_book()
                        
                        if new_max < current_max:
                            current_max = new_max
                            if current_max < best_max:
                                best_max = current_max
                                best_adj = adj.copy()
                            no_improve = 0
                        else:
                            remove_edge(ax, ay)
                            add_edge(rem_u, rem_v)
                            no_improve += 1
                            
                if no_improve > max_no_improve * 2:
                    break # Stuck, stop searching this instance
                    
    # Final check
    if best_max > current_max:
        best_adj = adj.copy()
        best_max = current_max

    # Convert best_adj to list of edges
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if has_edge(i, j) and (best_adj[i] >> j) & 1:
                # Wait, best_adj is a copy of adj.
                # We should use best_adj to extract edges.
                pass
    
    # Re-extract from best_adj
    result = []
    for i in range(n):
        for j in range(i+1, n):
            if (best_adj[i] >> j) & 1:
                result.append([i, j])
                
    assert len(result) == 401, f"Expected 401 edges, got {len(result)}"
    return result
