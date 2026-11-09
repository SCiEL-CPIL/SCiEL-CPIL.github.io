import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 40
    target_edges = 401
    
    # Use numpy adjacency matrix for fast book size computation
    # adj[i,j] = 1 if edge exists, 0 otherwise
    adj = np.zeros((n, n), dtype=np.int32)
    
    def add_edge(u, v):
        adj[u, v] = 1
        adj[v, u] = 1
    
    def remove_edge(u, v):
        adj[u, v] = 0
        adj[v, u] = 0
    
    def has_edge(u, v):
        return adj[u, v] == 1
    
    def book_size(u, v):
        """Number of common neighbors of u and v (triangles containing edge uv)"""
        if not has_edge(u, v):
            return 0
        # Common neighbors = sum of adj[u] * adj[v] excluding u and v themselves
        # adj[u] has 1 at position v, adj[v] has 1 at position u
        # So sum(adj[u] * adj[v]) counts common neighbors including u,v if self-loops
        # But we have no self-loops, so just count positions where both have 1
        return int(np.dot(adj[u], adj[v]))
    
    def max_book():
        """Find the maximum book size over all edges"""
        max_b = 0
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i, j]:
                    b = book_size(i, j)
                    if b > max_b:
                        max_b = b
        return max_b
    
    # Initialization: Start with K_{20,20} (400 edges) + 1 intra-partition edge
    # Then use local search to reduce the high book size
    A = list(range(20))
    B = list(range(20, 40))
    
    for a in A:
        for b in B:
            add_edge(a, b)
    
    # Add 1 edge within A - this creates book size 20 for that edge
    add_edge(0, 1)
    
    # Verify we have exactly 401 edges
    edge_count = int(np.sum(adj) // 2)
    assert edge_count == target_edges, f"Expected {target_edges}, got {edge_count}"
    
    # Local search with simulated annealing
    current_max = max_book()
    best_max = current_max
    best_adj = adj.copy()
    
    # Temperature schedule
    initial_temp = 10.0
    final_temp = 0.1
    cooling_rate = 0.999
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 500
    
    # Precompute all possible non-edges for faster sampling
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    while time.time() < deadline:
        # Find an edge with high book size to target
        # Sample a random edge
        u, v = random.choice(all_pairs)
        if not has_edge(u, v):
            # Pick a random edge instead
            for _ in range(10):
                u, v = random.choice(all_pairs)
                if has_edge(u, v):
                    break
            else:
                continue
        
        # Try to swap (u,v) with a random non-edge (x,y)
        x, y = random.choice(all_pairs)
        if has_edge(x, y) or x == u and y == v or x == v and y == u:
            continue
        
        # Compute change in book sizes
        # Removing (u,v) affects book sizes of edges sharing u or v
        # Adding (x,y) affects book sizes of edges sharing x or y
        
        # Quick check: what's the book size of (u,v) currently?
        b_uv = book_size(u, v)
        
        # What would be the book size of (x,y) if we added it?
        # It's the number of common neighbors of x and y
        b_xy = int(np.dot(adj[x], adj[y]))
        
        # If b_xy >= current_max and b_xy > b_uv, this swap might not help
        # But with annealing, we might still accept it
        
        # Perform the swap temporarily
        remove_edge(u, v)
        add_edge(x, y)
        
        # Recompute max book (expensive but necessary for correctness)
        new_max = max_book()
        
        # Accept or reject based on simulated annealing
        if new_max < current_max:
            accept = True
        elif new_max == current_max:
            accept = True
        else:
            # Accept with probability exp(-(new_max - current_max) / temp)
            delta = new_max - current_max
            prob = np.exp(-delta / temp) if delta > 0 else 1.0
            accept = random.random() < prob
        
        if accept:
            current_max = new_max
            no_improve_count = 0
            if new_max < best_max:
                best_max = new_max
                best_adj = adj.copy()
        else:
            # Revert
            remove_edge(x, y)
            add_edge(u, v)
            no_improve_count += 1
        
        # Cool down
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
        
        # If stuck for too long, restart with perturbation
        if no_improve_count > max_no_improve:
            # Perturb: randomly swap 5 edges
            for _ in range(5):
                u1, v1 = random.choice(all_pairs)
                if not has_edge(u1, v1):
                    continue
                u2, v2 = random.choice(all_pairs)
                if has_edge(u2, v2) or (u2, v2) == (u1, v1) or (u2, v2) == (v1, u1):
                    continue
                remove_edge(u1, v1)
                add_edge(u2, v2)
            current_max = max_book()
            no_improve_count = 0
            temp = initial_temp * 0.5  # Reset to medium temperature
    
    # Return the best solution found
    adj = best_adj
    result = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j]:
                result.append([i, j])
    
    return result
